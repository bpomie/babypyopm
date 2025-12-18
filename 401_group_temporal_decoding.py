#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 2025

Group-Level Temporal Decoding: Frequent vs Infrequent Tones
With Individual Subject Plots and Colored Group Plot

This script performs temporal decoding across multiple participants,
creates individual plots for each subject, and a group plot with
different colored lines per subject.

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import json

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_3_filter_ica_manualclean'
task = 'oddballTones'
fif_file_coda = '_file-oddballTones_processed_2_filter_ica.fif'

# Set to True to drop bad epochs from *_epochs_bad.txt file, False to keep all epochs
DROP_BAD_EPOCHS = True

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'


# Choose which contrast to decode:
#   'freq_vs_infreq' - Frequent vs Infrequent tones
#   'high_vs_low'    - High vs Low pitch tones

DECODING_CONTRAST = 'high_vs_low'  # <-- CHANGE THIS TO SWITCH CONTRAST

# Define contrast configurations
CONTRAST_CONFIG = {
    'freq_vs_infreq': {
        'condition_1': 'freq/tone',
        'condition_2': 'infreq/tone',
        'label_1': 'Frequent',
        'label_2': 'Infrequent',
        'title': 'Frequent vs Infrequent Tones',
        'folder_name': 'freq_vs_infreq'
    },
    'high_vs_low': {
        'condition_1': 'high',
        'condition_2': 'low',
        'label_1': 'High Pitch',
        'label_2': 'Low Pitch',
        'title': 'High vs Low Pitch Tones',
        'folder_name': 'high_vs_low'
    }
}

# Get current contrast configuration
contrast = CONTRAST_CONFIG[DECODING_CONTRAST]

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

# Create results directories (with contrast-specific subfolder)
path_results_decoding = os.path.join(
    root_data_path, 'results', 'decoding', input_folder, contrast['folder_name']
)
path_results_group = os.path.join(path_results_decoding, 'decoding_group')
path_results_individual = os.path.join(path_results_decoding, 'decoding_individual')
os.makedirs(path_results_group, exist_ok=True)
os.makedirs(path_results_individual, exist_ok=True)

# List all subjects
subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")
print(f"\nDecoding contrast: {contrast['title']}")

# For testing, you can limit to specific subjects:
# subjects = ['sub-101', 'sub-102', 'sub-105', 'sub-107']

# =============================================================================
# DECODING FUNCTION
# =============================================================================

def decode_participant(subj, paths, contrast):
    """
    Perform temporal decoding for a single participant.
    
    Parameters
    ----------
    subj : str
        Subject ID (e.g., 'sub-101')
    paths : object
        Paths object from utils
    contrast : dict
        Contrast configuration dictionary
        
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
        paths.data, subj, input_folder, f"{subj}" + fif_file_coda
    )
    path_bad_channels = os.path.join(paths.data, subj, f"{subj}_badchannels.tsv")
    path_event_dict = os.path.join(paths.data, subj, f"{subj}_event_dict.json")
    path_bad_epochs = os.path.join(paths.data, subj, f"{subj}_epochs_bad.txt")
    
    # Check if files exist
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
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=-0.1,
            tmax=0.5,
            baseline=None,
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
                    # Filter out indices that are out of range
                    valid_indices = [idx for idx in bad_epochs_indices if idx < len(epochs)]
                    if len(valid_indices) > 0:
                        print(f"Dropping {len(valid_indices)} bad epochs")
                        epochs.drop(valid_indices, reason='USER')
                    if len(valid_indices) < len(bad_epochs_indices):
                        print(f"  (Skipped {len(bad_epochs_indices) - len(valid_indices)} out-of-range indices)")
            except Exception as e:
                print(f"Warning: Could not load bad epochs: {e}")
        
        # Select conditions based on contrast
        try:
            epochs_cond1 = epochs[contrast['condition_1']]
            epochs_cond2 = epochs[contrast['condition_2']]
        except KeyError as e:
            print(f"WARNING: Condition {e} not found in event_dict for {subj}")
            print(f"Available conditions: {list(event_dict.keys())}")
            return None, None
        
        print(f"{contrast['label_1']}: {len(epochs_cond1)} epochs")
        print(f"{contrast['label_2']}: {len(epochs_cond2)} epochs")
        
        # Equalize epoch counts
        min_len = min(len(epochs_cond1), len(epochs_cond2))
        
        if min_len < 20:
            print(f"WARNING: Too few epochs ({min_len}) for {subj}")
            return None, None
        
        # Random sampling to equalize
        np.random.seed(42)
        idx_cond1 = np.random.choice(len(epochs_cond1), min_len, replace=False)
        idx_cond2 = np.random.choice(len(epochs_cond2), min_len, replace=False)
        
        epochs_equalized = mne.concatenate_epochs([
            epochs_cond1[idx_cond1],
            epochs_cond2[idx_cond2]
        ])
        
        print(f"Equalized to {min_len} epochs per condition")
        
        # Prepare data
        X = epochs_equalized.get_data(picks='mag')
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
        
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


def plot_individual_decoding(subj, scores, times, save_path, contrast):
    """
    Create and save individual subject decoding plot.
    
    Parameters
    ----------
    subj : str
        Subject ID
    scores : array
        CV scores (n_folds, n_times)
    times : array
        Time points
    save_path : str
        Directory to save figure
    contrast : dict
        Contrast configuration dictionary
    """
    
    mean_scores = np.mean(scores, axis=0)
    sem_scores = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reference lines
    ax.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance')
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Plot mean with SEM shading
    ax.plot(times, mean_scores, linewidth=2, color='#2E86AB', label='Mean AUC')
    ax.fill_between(
        times,
        mean_scores - sem_scores,
        mean_scores + sem_scores,
        alpha=0.3,
        color='#2E86AB',
        label='±SEM'
    )
    
    # Find and annotate peak
    peak_idx = np.argmax(mean_scores)
    peak_time = times[peak_idx]
    peak_score = mean_scores[peak_idx]
    ax.scatter([peak_time], [peak_score], color='#E94F37', s=80, zorder=5, 
               label=f'Peak: {peak_score:.3f} at {peak_time:.3f}s')
    
    # Dynamic y-axis limits with padding
    y_min = min(mean_scores.min() - sem_scores.max(), 0.35)
    y_max = max(mean_scores.max() + sem_scores.max(), 0.65)
    padding = (y_max - y_min) * 0.1
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title(f'Temporal Decoding: {subj}\n{contrast["title"]}', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.set_xlim([times[0], times[-1]])
    
    plt.tight_layout()
    
    # Save with contrast-specific filename
    fig_path = os.path.join(save_path, f'{subj}_temporal_decoding_{contrast["folder_name"]}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved individual plot: {fig_path}")
    
    plt.close(fig)


# =============================================================================
# PROCESS ALL PARTICIPANTS
# =============================================================================

all_scores = []
all_times = []
successful_subjects = []

for subj in subjects:
    scores, times = decode_participant(subj, paths, contrast)
    
    if scores is not None:
        # Save individual plot
        plot_individual_decoding(subj, scores, times, path_results_individual, contrast)
        
        # Store for group analysis
        all_scores.append(np.mean(scores, axis=0))
        all_times.append(times)
        successful_subjects.append(subj)

print(f"\n{'='*60}")
print(f"Successfully processed {len(successful_subjects)}/{len(subjects)} subjects")
print('='*60)

if len(all_scores) == 0:
    print("No subjects successfully processed!")
    exit()

# =============================================================================
# VERIFY TIME CONSISTENCY
# =============================================================================

time_lengths = [len(t) for t in all_times]
if len(set(time_lengths)) > 1:
    print("WARNING: Different time vectors across subjects!")
    print(f"Time lengths: {time_lengths}")
    min_time_len = min(time_lengths)
    all_scores = [s[:min_time_len] for s in all_scores]
    all_times = [t[:min_time_len] for t in all_times]

times = all_times[0]
scores_array = np.array(all_scores)

print(f"\nScores array shape: {scores_array.shape}")
print(f"Time points: {len(times)}")

# =============================================================================
# COMPUTE GROUP STATISTICS
# =============================================================================

mean_scores_group = np.mean(scores_array, axis=0)
sem_scores_group = np.std(scores_array, axis=0) / np.sqrt(len(all_scores))

# =============================================================================
# PLOT GROUP RESULTS - COLORED LINES PER SUBJECT
# =============================================================================

# Create colormap for individual subjects
colors = cm.tab10(np.linspace(0, 1, len(successful_subjects)))

fig, ax = plt.subplots(figsize=(12, 7))

# Reference lines
ax.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance', zorder=1)
ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

# Plot individual subjects with different colors
for i, subj in enumerate(successful_subjects):
    ax.plot(
        times,
        scores_array[i],
        alpha=0.6,
        linewidth=1.5,
        color=colors[i],
        label=subj,
        zorder=2
    )

# Plot grand average with thicker black line
ax.plot(
    times,
    mean_scores_group,
    linewidth=3,
    color='black',
    label=f'Grand Average (n={len(successful_subjects)})',
    zorder=3
)

# Add SEM shading for grand average
ax.fill_between(
    times,
    mean_scores_group - sem_scores_group,
    mean_scores_group + sem_scores_group,
    alpha=0.2,
    color='black',
    zorder=2
)

# Dynamic y-axis limits based on actual data range
y_min = scores_array.min()
y_max = scores_array.max()
y_range = y_max - y_min
padding = y_range * 0.15  # 15% padding on each side

# Ensure chance level (0.5) is visible and has some context
y_min_plot = min(y_min - padding, 0.45)
y_max_plot = max(y_max + padding, 0.55)

# Formatting
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('AUC Score', fontsize=14)
ax.set_title(
    f'Group Temporal Decoding: {contrast["title"]}',
    fontsize=16,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.set_ylim([y_min_plot, y_max_plot])
ax.set_xlim([times[0], times[-1]])

plt.tight_layout()

# Save figure with contrast-specific filename
fig_path = os.path.join(path_results_group, f'group_temporal_decoding_{contrast["folder_name"]}.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved group figure to: {fig_path}")

plt.show()

# =============================================================================
# STATISTICS SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"GROUP DECODING SUMMARY: {contrast['title']}")
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
print(f"\nMean AUC (post-stimulus): {mean_post_stim:.3f}")

# Baseline performance
baseline_mask = times < 0
mean_baseline = mean_scores_group[baseline_mask].mean()
print(f"Mean AUC (baseline): {mean_baseline:.3f}")

# Subject-wise statistics
print(f"\nSubject-wise peak AUC:")
subject_peaks = np.max(scores_array, axis=1)
for i, subj in enumerate(successful_subjects):
    peak_t = times[np.argmax(scores_array[i])]
    print(f"  {subj}: {subject_peaks[i]:.3f} at {peak_t:.3f}s")

print(f"\n  Mean: {np.mean(subject_peaks):.3f}")
print(f"  Std: {np.std(subject_peaks):.3f}")
print(f"  Range: [{np.min(subject_peaks):.3f}, {np.max(subject_peaks):.3f}]")

print(f"\n{'='*60}")

# =============================================================================
# SAVE GROUP RESULTS
# =============================================================================

# Save as CSV with contrast-specific filename
results_df = pd.DataFrame({
    'time': times,
    'mean_auc': mean_scores_group,
    'sem_auc': sem_scores_group
})

for i, subj in enumerate(successful_subjects):
    results_df[subj] = scores_array[i]

csv_path = os.path.join(path_results_group, f'group_decoding_results_{contrast["folder_name"]}.csv')
results_df.to_csv(csv_path, index=False)

# Save as numpy with contrast-specific filename
results_dict = {
    'times': times,
    'scores_array': scores_array,
    'mean_scores_group': mean_scores_group,
    'sem_scores_group': sem_scores_group,
    'subjects': successful_subjects,
    'n_subjects': len(successful_subjects),
    'contrast': contrast
}

np.save(
    os.path.join(path_results_group, f'group_decoding_results_{contrast["folder_name"]}.npy'),
    results_dict,
    allow_pickle=True
)

print(f"\nSaved group results to: {path_results_group}")
print(f"Individual plots saved to: {path_results_individual}")
print(f"\nContrast: {contrast['title']}")
print("\nGroup decoding analysis complete!")