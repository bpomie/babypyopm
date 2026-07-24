#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 2025

Group-Level Temporal Decoding: Frequent vs Infrequent Tones

This script performs temporal decoding across multiple participants
and creates grand average decoding performance curves.

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mne
# Pre-import to avoid circular import issue in MNE 1.10.0
import mne.stats.cluster_level
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import json
import sys  # For sys.exit()

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_2_filter_ica'
file_suffix='_file-oddballTones_processed_2_filter_ica'
task = 'oddballTones'

# Set to True to drop bad epochs from *_epochs_bad.txt file, False to keep all epochs
DROP_BAD_EPOCHS = True

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

# Create results directory
path_results_decoding = os.path.join(root_data_path, 'results', input_folder, 'decoding')

os.makedirs(path_results_decoding, exist_ok=True)

# List all subjects
subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")

# For testing, you can limit to specific subjects:
# subjects = ['sub-101', 'sub-102', 'sub-105', 'sub-107']

# =============================================================================
# DECODING FUNCTION
# =============================================================================

def decode_participant(subj, paths):
    """
    Perform temporal decoding for a single participant.
    
    Parameters
    ----------
    subj : str
        Subject ID (e.g., 'sub-101')
    paths : object
        Paths object from utils
        
    Returns
    -------
    scores : array
        Cross-validation scores (n_folds, n_times)
    times : array
        Time points
    """
    
    print(f"\n{'='*60}")
    print(f"Processing {subj}")
    print('='*60)
    
    # Construct paths
    path_task_data = os.path.join(
        paths.data, subj, input_folder, f"{subj}_file-{task}_processed_2_filter_ica.fif"
    )
    path_bad_channels = os.path.join(paths.data, subj, f"{subj}_badchannels.tsv")
    path_event_dict = os.path.join(paths.data, subj, f"{subj}_event_dict.json")
    path_bad_epochs = os.path.join(paths.data, subj, f"{subj}_epochs_bad.txt")
    
    # Check if files exist
    path_task_data_folder = os.path.join(
        paths.data, subj, input_folder
    )
    if not os.path.exists(path_task_data):
        print(f"WARNING: Data file not found for {subj}")
        return None, None
    
    try:
        # Load data
        raw = mne.io.read_raw_fif(path_task_data, preload=True, verbose=False)
        
        # Load and set bad channels
        if os.path.exists(path_bad_channels):
            bad_channels = pd.read_csv(path_bad_channels, sep='\t')
            bad_channels = bad_channels['badchannelslots'].tolist()
            raw.info['bads'].clear()
            raw.info['bads'].extend(bad_channels)
            print(f"Bad channels: {bad_channels}")
        
        # Load events
        events = mne.find_events(raw, stim_channel='di32', verbose=False)
        
        # Load event dictionary
        with open(path_event_dict, 'r') as f:
            event_dict = json.load(f)
        
        # Create epochs
        # No baseline correction - let the classifier learn from raw data
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=-0.1,
            tmax=0.5,
            baseline=(None, 0),  # No baseline correction for decoding
            detrend=1,
            reject_by_annotation=True,
            preload=True,
            verbose=False
        )
        
        # Drop bad epochs if enabled and file exists
        if DROP_BAD_EPOCHS and os.path.exists(path_bad_epochs):
            try:
                with open(path_bad_epochs, 'r') as f:
                    bad_epochs_list = [line.strip() for line in f if line.strip()]
                bad_epochs_indices = [int(idx) for idx in bad_epochs_list]
                
                if len(bad_epochs_indices) > 0:
                    print(f"Dropping {len(bad_epochs_indices)} bad epochs")
                    epochs.drop(bad_epochs_indices, reason='USER')
            except Exception as e:
                print(f"Warning: Could not load bad epochs: {e}")
        
        # Select conditions
        epochs_freq = epochs["freq/tone"]
        epochs_infreq = epochs["infreq/tone"]
        
        print(f"Frequent: {len(epochs_freq)} epochs")
        print(f"Infrequent: {len(epochs_infreq)} epochs")
        
        # Check minimum epoch count before equalization
        min_len = min(len(epochs_freq), len(epochs_infreq))
        
        if min_len < 20:  # Minimum threshold
            print(f"WARNING: Too few epochs ({min_len}) for {subj}")
            return None, None
        
        # Use MNE's built-in equalization function
        mne.epochs.equalize_epoch_counts([epochs_freq, epochs_infreq], method='mintime')
        
        print(f"Equalized to {len(epochs_freq)} epochs per condition")
        
        # Combine epochs
        epochs_equalized = mne.concatenate_epochs([epochs_freq, epochs_infreq])
        
        # Prepare data
        X = epochs_equalized.get_data(picks='mag')
        y = np.concatenate([np.zeros(len(epochs_freq)), np.ones(len(epochs_infreq))])
        
        # Set up classifier
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear', random_state=42)
        )
        
        time_decoder = SlidingEstimator(
            clf,
            n_jobs=1,
            scoring='roc_auc',
            verbose=False
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Decode
        scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)
        times = epochs_equalized.times
        
        mean_auc = np.mean(scores)
        print(f"Mean AUC: {mean_auc:.3f}")
        print(f"Peak AUC: {np.max(np.mean(scores, axis=0)):.3f}")
        
        return scores, times
        
    except Exception as e:
        print(f"ERROR processing {subj}: {str(e)}")
        return None, None

# =============================================================================
# PROCESS ALL PARTICIPANTS
# =============================================================================

all_scores = []
all_times = []
successful_subjects = []

for subj in subjects:
    scores, times = decode_participant(subj, paths)
    
    if scores is not None:
        all_scores.append(np.mean(scores, axis=0))  # Average across CV folds
        all_times.append(times)
        successful_subjects.append(subj)

print(f"\n{'='*60}")
print(f"Successfully processed {len(successful_subjects)}/{len(subjects)} subjects")
print('='*60)

if len(all_scores) == 0:
    print("No subjects successfully processed!")
    sys.exit()

# =============================================================================
# VERIFY TIME CONSISTENCY
# =============================================================================

# Check that all time vectors are the same
time_lengths = [len(t) for t in all_times]
if len(set(time_lengths)) > 1:
    print("WARNING: Different time vectors across subjects!")
    print(f"Time lengths: {time_lengths}")
    # Trim to shortest
    min_time_len = min(time_lengths)
    all_scores = [s[:min_time_len] for s in all_scores]
    all_times = [t[:min_time_len] for t in all_times]

# Use first subject's times (they should all be the same)
times = all_times[0]

# Convert to array: (n_subjects, n_times)
scores_array = np.array(all_scores)

print(f"\nScores array shape: {scores_array.shape}")
print(f"Time points: {len(times)}")

# =============================================================================
# COMPUTE GROUP STATISTICS
# =============================================================================

# Mean and SEM across subjects
mean_scores_group = np.mean(scores_array, axis=0)
sem_scores_group = np.std(scores_array, axis=0) / np.sqrt(len(all_scores))

# =============================================================================
# PLOT GROUP RESULTS
# =============================================================================

# Create a colormap for individual subjects
import matplotlib.cm as cm
colors = cm.tab10(np.linspace(0, 1, len(successful_subjects)))

fig, ax = plt.subplots(figsize=(12, 7))

# Add chance level and stimulus onset
ax.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance', zorder=1)
ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

# Plot individual subjects with different colors
for i, subj in enumerate(successful_subjects):
    ax.plot(
        times,
        scores_array[i],
        alpha=0.5,
        linewidth=1.5,
        color=colors[i],
        label=subj,
        zorder=2
    )

# Plot grand average with thicker line and shaded SEM
ax.plot(
    times,
    mean_scores_group,
    linewidth=3,
    color='black',
    label=f'Grand Average (n={len(successful_subjects)})',
    zorder=3
)

ax.fill_between(
    times,
    mean_scores_group - sem_scores_group,
    mean_scores_group + sem_scores_group,
    alpha=0.2,
    color='black',
    zorder=2
)

# Formatting
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('AUC Score', fontsize=14)
ax.set_title(
    'Group Temporal Decoding: Frequent vs Infrequent Tones',
    fontsize=16,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(False)
ax.set_ylim([0.37, 0.7])

plt.tight_layout()

# Save figure
fig_path = os.path.join(path_results_decoding, 'group_temporal_decoding.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved group figure to: {fig_path}")

plt.show()

# =============================================================================
# STATISTICS SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print("GROUP DECODING SUMMARY")
print('='*60)

# Peak decoding
peak_idx = np.argmax(mean_scores_group)
peak_time = times[peak_idx]
peak_score = mean_scores_group[peak_idx]

print(f"\nPeak AUC: {peak_score:.3f} at {peak_time:.3f} s")
print(f"SEM at peak: {sem_scores_group[peak_idx]:.3f}")

# Post-stimulus performance
post_stim_mask = times >= 0
mean_post_stim = mean_scores_group[post_stim_mask].mean()
print(f"\nMean AUC (0-0.5s): {mean_post_stim:.3f}")

# Baseline performance
baseline_mask = times < 0
mean_baseline = mean_scores_group[baseline_mask].mean()
print(f"Mean AUC (baseline): {mean_baseline:.3f}")

# Subject-wise statistics
print(f"\nSubject-wise peak AUC:")
subject_peaks = np.max(scores_array, axis=1)
print(f"  Mean: {np.mean(subject_peaks):.3f}")
print(f"  Std: {np.std(subject_peaks):.3f}")
print(f"  Range: [{np.min(subject_peaks):.3f}, {np.max(subject_peaks):.3f}]")

print(f"\n{'='*60}")

# =============================================================================
# SAVE GROUP RESULTS
# =============================================================================

# Save group decoding time series to CSV
results_df = pd.DataFrame({
    'time': times,
    'mean_auc': mean_scores_group,
    'sem_auc': sem_scores_group
})

# Add individual subject columns
for i, subj in enumerate(successful_subjects):
    results_df[subj] = scores_array[i]

csv_path = os.path.join(path_results_decoding, 'group_decoding_timeseries.csv')
results_df.to_csv(csv_path, index=False)

print(f"\nSaved group decoding time series to: {csv_path}")
print(f"Number of subjects: {len(successful_subjects)}")
print("\nGroup decoding analysis complete!")