#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
400_temporal_decoding.py


Workflow:
  1. Decode each participant (sliding estimator, 5-fold CV, AUC), caching the
     data needed for pattern extraction so nothing is decoded twice
  2. Individual subject plots
  3. Group colored plot + statistics summary + CSV/NPY saves
  4. Peak identification, binomial test, MVPA summary text file
  5. Discriminant pattern extraction at the group peak (from cached data)
  6. Combined figure:
       Left panel  - smoothed temporal decoding curves with peak annotation
       Right panel - per-subject discriminant activation topographies

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel, get_coef
from mne.viz import plot_topomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os
import pandas as pd
import json

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_2_filter_ica'
task = 'oddballTones'
fif_file_coda = '_file-oddballTones_preprocessing_routine_3.fif'

# Set to True to drop bad epochs from *_epochs_bad.txt file, False to keep all epochs
DROP_BAD_EPOCHS = True

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# Figure parameters (combined publication figure)
SMOOTHING_SIGMA = 10  # Gaussian smoothing for visualization only (0 = off)
FIGURE_DPI = 300
SAVE_FIGURES = True

# Color scheme for subjects in the publication figure (up to 8)
SUBJECT_COLORS = [
    '#9467BD',  # purple
    '#4DBBD5',  # cyan
    '#00A087',  # teal
    '#3C5488',  # navy
    '#E377C2',  # pink
    '#8491B4',  # slate
    '#91D1C2',  # mint
    '#D86C9E',  # magenta-pink
]

# Choose which contrast to decode:
#   'freq_vs_infreq' - Frequent vs Infrequent tones
#   'high_vs_low'    - High vs Low pitch tones

DECODING_CONTRAST = 'freq_vs_infreq'  # <-- CHANGE THIS TO SWITCH CONTRAST

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
path_results_figure = os.path.join(path_results_decoding, 'combined_figure')
os.makedirs(path_results_group, exist_ok=True)
os.makedirs(path_results_individual, exist_ok=True)
os.makedirs(path_results_figure, exist_ok=True)

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

    Also returns the (channels-cleaned) data matrix, labels, and channel info
    so that discriminant patterns can later be extracted at the group peak
    without re-loading or re-decoding.

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
    X : array
        Data matrix (n_epochs, n_channels, n_times), mag channels, bads excluded
    y : array
        Labels (0 = condition_1, 1 = condition_2)
    info : mne.Info
        Channel info matching X (for topomap plotting)
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
        return None, None, None, None, None

    try:
        # Load data
        raw = mne.io.read_raw_fif(path_task_data, preload=True, verbose=False)

        # Load and set bad channels
        if os.path.exists(path_bad_channels):
            bad_channels = pd.read_csv(path_bad_channels, sep='\t')
            bad_channels = bad_channels['badchannelslots'].tolist()

            # Clean up entries (drop NaNs, strip whitespace)
            bad_channels = [str(ch).strip() for ch in bad_channels if pd.notna(ch)]

            # Keep only bad channels that actually exist in this recording
            # (guards against naming-convention mismatches, e.g. 's24_bz' vs 'L11')
            missing_bads = [ch for ch in bad_channels if ch not in raw.ch_names]
            bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

            if missing_bads:
                print(f"WARNING: bad channels not found in data for {subj}, skipped: {missing_bads}")

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
            tmax=0.6,
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
            return None, None, None, None, None

        print(f"{contrast['label_1']}: {len(epochs_cond1)} epochs")
        print(f"{contrast['label_2']}: {len(epochs_cond2)} epochs")

        # Equalize epoch counts
        min_len = min(len(epochs_cond1), len(epochs_cond2))

        if min_len < 20:
            print(f"WARNING: Too few epochs ({min_len}) for {subj}")
            return None, None, None, None, None

        # Random sampling to equalize
        np.random.seed(42)
        idx_cond1 = np.random.choice(len(epochs_cond1), min_len, replace=False)
        idx_cond2 = np.random.choice(len(epochs_cond2), min_len, replace=False)

        epochs_equalized = mne.concatenate_epochs([
            epochs_cond1[idx_cond1],
            epochs_cond2[idx_cond2]
        ])

        print(f"Equalized to {min_len} epochs per condition")

        # Prepare data - explicitly exclude bad channels (needed for topomaps)
        epochs_mag = epochs_equalized.copy().pick('mag', exclude='bads')

        n_channels = len(epochs_mag.ch_names)
        n_bads = len(epochs_equalized.info['bads'])
        print(f"Using {n_channels} channels (excluded {n_bads} bad)")

        X = epochs_mag.get_data(copy=True)
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
        times = epochs_mag.times
        info = epochs_mag.info.copy()

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

        mean_auc = np.mean(scores)
        print(f"Mean AUC: {mean_auc:.3f}")
        print(f"Peak AUC: {np.max(np.mean(scores, axis=0)):.3f}")

        return scores, times, X, y, info

    except Exception as e:
        print(f"ERROR processing {subj}: {str(e)}")
        return None, None, None, None, None


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


def identify_peak(mean_scores, times, min_time=0.0, prominence=0.01, distance=15):
    """Identify the largest peak in the mean decoding curve (post-stimulus)."""
    post_stim_mask = times >= min_time
    post_stim_scores = mean_scores.copy()
    post_stim_scores[~post_stim_mask] = 0

    peak_indices, _ = find_peaks(post_stim_scores, prominence=prominence, distance=distance)

    if len(peak_indices) > 1:
        largest_idx = np.argmax(mean_scores[peak_indices])
        peak_indices = np.array([peak_indices[largest_idx]])

    return peak_indices[0] if len(peak_indices) > 0 else np.argmax(post_stim_scores)


# =============================================================================
# PROCESS ALL PARTICIPANTS
# =============================================================================

all_scores = []
all_times = []
successful_subjects = []

# Cached per-subject data for pattern extraction at the group peak
all_X = {}
all_y = {}
all_info = {}
all_subject_times = {}

for subj in subjects:
    scores, times, X, y, info = decode_participant(subj, paths, contrast)

    if scores is not None:
        # Save individual plot
        plot_individual_decoding(subj, scores, times, path_results_individual, contrast)

        # Store for group analysis
        all_scores.append(np.mean(scores, axis=0))
        all_times.append(times)
        successful_subjects.append(subj)

        # Cache for pattern extraction
        all_X[subj] = X
        all_y[subj] = y
        all_info[subj] = info
        all_subject_times[subj] = times

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

# Peak decoding (find_peaks on post-stimulus window; falls back to argmax)
peak_idx = identify_peak(mean_scores_group, times)
peak_time = times[peak_idx]
peak_value = mean_scores_group[peak_idx]
scores_at_peak = scores_array[:, peak_idx]

print(f"\nPeak AUC: {peak_value:.3f} at {peak_time:.3f} s")
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
# BINOMIAL TEST & MVPA SUMMARY
# =============================================================================

n_subjects = len(successful_subjects)
n_above_chance = np.sum(scores_at_peak > 0.5)
binomial_result = binomtest(n_above_chance, n_subjects, p=0.5, alternative='greater')

print(f"Binomial test: {n_above_chance}/{n_subjects} above chance at group peak, "
      f"p = {binomial_result.pvalue:.4f}")

# Print MVPA results summary for paper
print("\n" + "-"*60)
print("MVPA RESULTS SUMMARY (for paper)")
print("-"*60)
print(f"Contrast: {contrast['title']}")
print(f"Temporal decoding was performed using a sliding estimator with")
print(f"logistic regression (5-fold cross-validation, AUC scoring).")
print(f"")
print(f"Peak decoding accuracy: AUC = {peak_value:.3f} at {peak_time*1000:.0f} ms")
print(f"Mean AUC across subjects at peak: {np.mean(scores_at_peak):.3f} ± {np.std(scores_at_peak):.3f}")
print(f"")
print(f"Given the small sample size (n={n_subjects}), formal statistical inference")
print(f"is limited. However, a binomial test at group peak decoding time showed")
print(f"that {n_above_chance}/{n_subjects} participants decoded above chance level")
print(f"(p = {binomial_result.pvalue:.3f}).")
print("-"*60)

# Save MVPA results summary to text file
mvpa_summary = f"""MVPA RESULTS SUMMARY
====================

Analysis: Temporal decoding of {contrast['title'].lower()}
Method: Sliding estimator with logistic regression
Cross-validation: 5-fold stratified
Scoring metric: Area Under the ROC Curve (AUC)

RESULTS
-------
Peak decoding time: {peak_time*1000:.0f} ms
Peak group AUC: {peak_value:.3f}
Mean AUC at peak (± SD): {np.mean(scores_at_peak):.3f} ± {np.std(scores_at_peak):.3f}

Individual subject AUCs at peak:
{chr(10).join([f"  {s.replace('sub-', 'S')}: {scores_at_peak[i]:.3f}" for i, s in enumerate(successful_subjects)])}

STATISTICAL NOTE
----------------
Given the small sample size (n={n_subjects}), formal statistical inference
is limited. However, a binomial test at group peak decoding time showed
that {n_above_chance}/{n_subjects} participants decoded above chance level
(AUC > 0.5), p = {binomial_result.pvalue:.3f}.
"""

mvpa_summary_path = os.path.join(path_results_group, f'mvpa_results_summary_{contrast["folder_name"]}.txt')
with open(mvpa_summary_path, 'w') as f:
    f.write(mvpa_summary)
print(f"\nSaved MVPA summary: {mvpa_summary_path}")

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

# Save peak summary
peak_df = pd.DataFrame({
    'subject': successful_subjects,
    'auc_at_peak': scores_at_peak,
    'above_chance': scores_at_peak > 0.5
})
peak_csv_path = os.path.join(path_results_group, f'peak_subject_scores_{contrast["folder_name"]}.csv')
peak_df.to_csv(peak_csv_path, index=False)
print(f"Saved peak scores: {peak_csv_path}")

# Save as numpy with contrast-specific filename
results_dict = {
    'times': times,
    'scores_array': scores_array,
    'mean_scores_group': mean_scores_group,
    'sem_scores_group': sem_scores_group,
    'subjects': successful_subjects,
    'n_subjects': len(successful_subjects),
    'contrast': contrast,
    'peak_time': peak_time,
    'peak_value': peak_value,
    'scores_at_peak': scores_at_peak,
    'binomial_pvalue': binomial_result.pvalue
}

np.save(
    os.path.join(path_results_group, f'group_decoding_results_{contrast["folder_name"]}.npy'),
    results_dict,
    allow_pickle=True
)

print(f"\nSaved group results to: {path_results_group}")
print(f"Individual plots saved to: {path_results_individual}")

# =============================================================================
# EXTRACT DISCRIMINANT PATTERNS AT GROUP PEAK
# =============================================================================

print(f"\nExtracting discriminant patterns at group peak ({peak_time*1000:.1f} ms)...")

all_patterns = {}

for subj in successful_subjects:
    X = all_X[subj]
    y = all_y[subj]
    subj_times = all_subject_times[subj]

    # Locate the group peak in this subject's time vector
    subj_peak_idx = np.argmin(np.abs(subj_times - peak_time))

    # Data at the peak time point
    X_peak = X[:, :, subj_peak_idx]

    # Fit LinearModel to recover interpretable activation patterns
    clf_pattern = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear', random_state=42))
    )
    clf_pattern.fit(X_peak, y)

    patterns = get_coef(clf_pattern, 'patterns_', inverse_transform=True)
    all_patterns[subj] = patterns

    print(f"  {subj}: patterns extracted ({len(patterns)} channels)")

# =============================================================================
# COMBINED PUBLICATION FIGURE (temporal decoding + topographies)
# =============================================================================

print("\n" + "="*70)
print("CREATING COMBINED PUBLICATION FIGURE")
print("="*70)

# Subject colors for the publication figure (fixed palette, cycled if > 8)
subject_color_map = {subj: SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
                     for i, subj in enumerate(successful_subjects)}

# Apply smoothing for visualization
if SMOOTHING_SIGMA > 0:
    mean_scores_smooth = gaussian_filter1d(mean_scores_group, sigma=SMOOTHING_SIGMA)
    sem_scores_smooth = gaussian_filter1d(sem_scores_group, sigma=SMOOTHING_SIGMA)
    scores_array_smooth = np.array([gaussian_filter1d(s, sigma=SMOOTHING_SIGMA) for s in scores_array])

    # Recalculate peak on smoothed data (for annotation placement only)
    post_stim_mask = times >= 0
    peak_idx_smooth = np.argmax(mean_scores_smooth * post_stim_mask)
    peak_time_smooth = times[peak_idx_smooth]
    peak_value_smooth = mean_scores_smooth[peak_idx_smooth]
else:
    mean_scores_smooth = mean_scores_group
    sem_scores_smooth = sem_scores_group
    scores_array_smooth = scores_array
    peak_time_smooth = peak_time
    peak_value_smooth = peak_value

# --- Create figure with custom layout ---
fig = plt.figure(figsize=(18, 10))

# GridSpec: left (temporal), right (topographies) - stretch temporal plot more
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.35)

# --- LEFT PANEL: Temporal Decoding ---
ax_temporal = fig.add_subplot(gs[0])

# Add frame
for spine in ax_temporal.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_edgecolor('black')

# Reference lines
ax_temporal.axhline(0.5, color='k', linestyle='--', linewidth=2.5, label='Chance', zorder=1)
ax_temporal.axvline(0, color='gray', linestyle='-', linewidth=2, alpha=0.5, zorder=1)

# Individual subjects
for i, subj in enumerate(successful_subjects):
    label = subj.replace('sub-', 'S')
    ax_temporal.plot(times * 1000, scores_array_smooth[i],
                     alpha=0.4, linewidth=2.5, color=subject_color_map[subj], label=label)

# Grand average with SEM
ax_temporal.plot(times * 1000, mean_scores_smooth, linewidth=4, color='black',
                 label='Grand Average', zorder=4)
ax_temporal.fill_between(
    times * 1000,
    mean_scores_smooth - sem_scores_smooth,
    mean_scores_smooth + sem_scores_smooth,
    alpha=0.2, color='black', zorder=3
)

# Peak annotation - positioned lower
ax_temporal.annotate(
    f'Group peak\n{peak_time*1000:.0f} ms\nAUC: {np.mean(scores_at_peak):.2f} ({np.std(scores_at_peak):.2f})',
    xy=(peak_time_smooth * 1000, peak_value_smooth),
    xytext=(peak_time_smooth * 1000, 0.38),
    fontsize=20, ha='center',
    arrowprops=dict(arrowstyle='->', color='black', lw=2.5),
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

ax_temporal.set_xlabel('Time (ms)', fontsize=24)
ax_temporal.set_ylabel('AUC', fontsize=24)
# Title added via fig.text for alignment with right panel
ax_temporal.legend(loc='upper right', fontsize=18, ncol=2)
ax_temporal.set_xlim([times[0]*1000, times[-1]*1000])
ax_temporal.set_ylim([0.35, 0.80])
ax_temporal.tick_params(axis='both', labelsize=20)

# Add title for temporal panel using fig.text for alignment
gs_pos_temporal = gs[0].get_position(fig)
fig.text((gs_pos_temporal.x0 + gs_pos_temporal.x1) / 2, 0.86,
         f'Temporal Decoding of\n{contrast["title"]}',
         fontsize=26, fontweight='bold', ha='center', va='bottom')

# --- RIGHT PANEL: Topographies (3 rows x 2 columns grid with colorbars) ---
# Order subjects by ID number (not by ranking)
ordered_subjects = sorted(successful_subjects)

# Create nested GridSpec for topographies with tighter spacing
n_topo_rows = 3
n_topo_cols = 2
gs_topo = gridspec.GridSpecFromSubplotSpec(
    n_topo_rows, n_topo_cols, subplot_spec=gs[1],
    wspace=0.6, hspace=0.55,
)

topo_axes = []
for idx, subj in enumerate(ordered_subjects):
    if idx >= n_topo_rows * n_topo_cols:  # Max topographies based on grid
        break

    row = idx // n_topo_cols
    col = idx % n_topo_cols

    ax_topo = fig.add_subplot(gs_topo[row, col])
    topo_axes.append(ax_topo)

    pattern_data = all_patterns[subj]
    info = all_info[subj]

    # Get sensor positions
    sensor_pos = mne.channels.layout._find_topomap_coords(info, picks='mag')

    # Plot - automatic scaling based on data range
    vmax = np.abs(pattern_data).max()
    vmin = -vmax

    im, _ = plot_topomap(
        pattern_data,
        sensor_pos,
        axes=ax_topo,
        show=False,
        vlim=(vmin, vmax),
        cmap='RdBu_r',
        sensors=False,  # Hide sensor markers for cleaner look
    )

    # Add colorbar - smaller bar but bigger text, more padding from topoplot
    divider = make_axes_locatable(ax_topo)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('a.u.', fontsize=18)

    # Title with subject ID in matching color - positioned higher
    short_label = subj.replace('sub-', 'S')
    ax_topo.set_title(short_label, fontsize=24, fontweight='bold',
                      color=subject_color_map[subj], pad=8, y=1.05)

# Add title for topo panel - aligned with temporal plot title
gs_pos = gs[1].get_position(fig)
fig.text((gs_pos.x0 + gs_pos.x1) / 2, 0.86,
         f'Discriminant Activation Patterns\n({peak_time*1000:.0f} ms)',
         fontsize=26, fontweight='bold', ha='center', va='bottom')

# Adjust subplot positions to make room for titles
fig.subplots_adjust(top=0.80)

# Save figure
if SAVE_FIGURES:
    fig_path = os.path.join(path_results_figure, f'combined_decoding_figure_{contrast["folder_name"]}.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {fig_path}")

    fig_pdf = os.path.join(path_results_figure, f'combined_decoding_figure_{contrast["folder_name"]}.pdf')
    fig.savefig(fig_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_pdf}")

plt.show()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"\nContrast: {contrast['title']}")