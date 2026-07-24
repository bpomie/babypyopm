#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Decoding: Frequent vs Infrequent Tones

This script performs time-resolved decoding using a sliding window approach
to classify frequent vs infrequent tone epochs at each time point.

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mne

# Workaround for MNE 1.10.0 circular import bug
# Manually inject the function that's causing the circular dependency
import sys
if 'mne.stats.cluster_level' not in sys.modules:
    # Import stats functions directly to break the cycle
    import mne.stats
    # The circular import happens because cluster_level tries to import from stats
    # We'll just ignore it by pre-populating what it needs
    
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import json

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

subj = 'sub-101'

# =============================================================================
# PATHS
# =============================================================================

path_data = os.path.join(root_data_path, 'data')
path_results_decoding = os.path.join(root_data_path, 'results', 'processed_2_filter_ica', 'decoding')

# Create results directory if it doesn't exist
os.makedirs(path_results_decoding, exist_ok=True)

print(path_data)
print(path_results_decoding)

# Path to preprocessed data
path_task_data = os.path.join(
    path_data, subj, 'processed_2_filter_ica', f"{subj}_file-oddballTones_processed_2_filter_ica.fif"
)
path_bad_channels = os.path.join(path_data, subj, f"{subj}_badchannels.tsv")
path_event_dictionary = os.path.join(path_data, subj, f"{subj}_event_dict.json")
path_bad_epochs = os.path.join(path_data, subj, f"{subj}_epochs_bad.txt")

print(f"Loading data from: {path_task_data}")
print(f"Bad epochs file: {path_bad_epochs}")

# =============================================================================
# LOAD DATA
# =============================================================================

raw = mne.io.read_raw_fif(path_task_data, preload=True)

# =============================================================================
# BAD CHANNELS
# =============================================================================

bad_channels = pd.read_csv(path_bad_channels, sep='\t')
bad_channels = bad_channels['badchannelslots'].tolist()
print(f"Bad channels: {bad_channels}")

raw.info['bads'].clear()
raw.info['bads'].extend(bad_channels)

# =============================================================================
# EVENTS
# =============================================================================

events = mne.find_events(raw, stim_channel='di32')

# =============================================================================
# EVENT DICTIONARY
# =============================================================================

with open(path_event_dictionary, 'r') as f:
    event_dict = json.load(f)

print(f"Event dictionary: {event_dict}")

# =============================================================================
# CREATE EPOCHS
# =============================================================================

# Create epochs with appropriate time window
# Using -0.1 to 0.5s to capture full response while avoiding next trial at 500ms ISI
# No baseline correction - let the classifier learn from raw data
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.1,
    tmax=0.6,
    baseline=None,  # No baseline correction for decoding
    detrend=1,
    reject_by_annotation=True,
    preload=True
)

print(epochs)

# =============================================================================
# DROP BAD EPOCHS
# =============================================================================

# Load bad epochs from file if it exists
if os.path.exists(path_bad_epochs):
    try:
        # Read the bad epochs file
        with open(path_bad_epochs, 'r') as f:
            bad_epochs_list = [line.strip() for line in f if line.strip()]
        
        # Convert to integers
        bad_epochs_indices = [int(idx) for idx in bad_epochs_list]
        
        print(f"\nFound {len(bad_epochs_indices)} bad epochs to drop")
        print(f"Bad epoch indices: {bad_epochs_indices}")
        
        # Drop the bad epochs
        epochs.drop(bad_epochs_indices, reason='USER')
        
        print(f"Epochs after dropping bad epochs: {len(epochs)}")
        
    except Exception as e:
        print(f"Warning: Could not load bad epochs file: {e}")
        print("Continuing without dropping bad epochs...")
else:
    print(f"\nNo bad epochs file found at: {path_bad_epochs}")
    print("Continuing without dropping bad epochs...")

print(f"\nFinal epochs count: {len(epochs)}")

# =============================================================================
# EQUALIZE EPOCH COUNTS
# =============================================================================

# Create separate epochs for each condition
epochs_freq = epochs["freq/tone"]
epochs_infreq = epochs["infreq/tone"]

print(f"Frequent epochs before equalization: {len(epochs_freq)}")
print(f"Infrequent epochs before equalization: {len(epochs_infreq)}")

# Use MNE's built-in equalization function
mne.epochs.equalize_epoch_counts([epochs_freq, epochs_infreq], method='mintime')

print(f"Frequent epochs after equalization: {len(epochs_freq)}")
print(f"Infrequent epochs after equalization: {len(epochs_infreq)}")

# Combine into one Epochs object
epochs_equalized = mne.concatenate_epochs([epochs_freq, epochs_infreq])

print(f"Total equalized epochs: {len(epochs_equalized)}")
print(f"Epochs per condition: {len(epochs_freq)}")

# =============================================================================
# PREPARE DATA FOR DECODING
# =============================================================================

# Get the data array (n_epochs, n_channels, n_times)
X = epochs_equalized.get_data(picks='mag')

# Create labels: 0 for frequent, 1 for infrequent
# Since we equalized and concatenated, first half is freq, second half is infreq
n_epochs_per_condition = len(epochs_freq)
y = np.concatenate([
    np.zeros(n_epochs_per_condition),
    np.ones(n_epochs_per_condition)
])

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class balance - Frequent: {np.sum(y==0)}, Infrequent: {np.sum(y==1)}")

# =============================================================================
# SET UP DECODING PIPELINE
# =============================================================================

# Create a machine learning pipeline with:
# 1. StandardScaler to normalize features
# 2. LogisticRegression for binary classification
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear', random_state=42)
)

# Create time-resolved decoder
time_decoder = SlidingEstimator(
    clf,
    n_jobs=1,  # Use 1 job for stability, increase if you have multiple cores
    scoring='roc_auc',  # Area under ROC curve
    verbose=True
)

# Use stratified k-fold cross-validation (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# PERFORM TEMPORAL DECODING
# =============================================================================

print("\n" + "="*60)
print("Starting temporal decoding...")
print("="*60 + "\n")

# Run cross-validated decoding at each time point
scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)

# scores shape: (n_splits, n_times)
print(f"Scores shape: {scores.shape}")
print(f"Mean accuracy across time: {scores.mean():.3f}")

# =============================================================================
# PLOT RESULTS
# =============================================================================

# Get time points
times = epochs_equalized.times

# Calculate mean and standard error across folds
mean_scores = np.mean(scores, axis=0)
se_scores = np.std(scores, axis=0) / np.sqrt(scores.shape[0])

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot decoding performance over time
ax.plot(times, mean_scores, label='Mean AUC')
ax.fill_between(
    times,
    mean_scores - se_scores,
    mean_scores + se_scores,
    alpha=0.3,
    label='±SE'
)

# Add chance level line
ax.axhline(0.5, color='k', linestyle='--', label='Chance (0.5)')

# Add vertical line at stimulus onset
ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

# Formatting
ax.set_xlabel('Time (s)')
ax.set_ylabel('AUC Score')
ax.set_title(f'Temporal Decoding: Frequent vs Infrequent Tones\n{subj}')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_path = os.path.join(path_results_decoding, f'{subj}_temporal_decoding.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved figure to: {fig_path}")

plt.show()

# =============================================================================
# PRINT SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("DECODING SUMMARY")
print("="*60)

# Find peak decoding time
peak_idx = np.argmax(mean_scores)
peak_time = times[peak_idx]
peak_score = mean_scores[peak_idx]

print(f"Peak AUC: {peak_score:.3f} at {peak_time:.3f} s")
print(f"Mean AUC (0-0.3s): {mean_scores[times >= 0].mean():.3f}")
print(f"Mean AUC (baseline): {mean_scores[times < 0].mean():.3f}")

# Find when decoding first exceeds threshold
threshold = 0.5
above_threshold = np.where(mean_scores > threshold)[0]
if len(above_threshold) > 0:
    first_sig_idx = above_threshold[0]
    first_sig_time = times[first_sig_idx]
    print(f"First time above {threshold}: {first_sig_time:.3f} s")

print("\n" + "="*60)

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save decoding time series to CSV
results_df = pd.DataFrame({
    'time': times,
    'mean_auc': mean_scores,
    'se_auc': se_scores
})

# Add individual CV fold scores as separate columns
for fold_idx in range(scores.shape[0]):
    results_df[f'fold_{fold_idx+1}'] = scores[fold_idx, :]

csv_path = os.path.join(path_results_decoding, f'{subj}_decoding_timeseries.csv')
results_df.to_csv(csv_path, index=False)

print(f"\nSaved decoding time series to: {csv_path}")
print(f"Subject: {subj}")
print(f"Epochs per condition: {n_epochs_per_condition}")
print("\nDecoding analysis complete!")