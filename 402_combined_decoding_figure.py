#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
402_combined_decoding_figure.py

Combined script that:
1. Computes temporal decoding (from 402a)
2. Extracts LinearModel patterns (from 402c)
3. Creates a 3-panel publication figure:
   - Left: Temporal decoding curves
   - Middle: Subject ranking by AUC at peak
   - Right: Topographical patterns (2x4 grid, ordered by rank)

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel, get_coef
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_2_filter_ica'
task = 'oddballTones'

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# Figure parameters
SMOOTHING_SIGMA = 10  # Gaussian smoothing for visualization
FIGURE_DPI = 300
SAVE_FIGURES = True

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

path_results = os.path.join(root_data_path, 'results', 'combined_decoding')
os.makedirs(path_results, exist_ok=True)

subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")

# Color scheme for subjects (up to 8)
SUBJECT_COLORS = [
    '#9467BD',  # purple (was red)
    '#4DBBD5',  # cyan
    '#00A087',  # teal
    '#3C5488',  # navy
    '#E377C2',  # pink
    '#8491B4',  # slate
    '#91D1C2',  # mint
    '#D86C9E',  # magenta-pink
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_epochs_for_subject(subj, paths):
    """Load and prepare epochs for a participant."""
    
    path_task_data = os.path.join(
        paths.data, subj, input_folder, f"{subj}_file-oddballTones_processed_2_filter_ica.fif"
    )
    path_bad_channels = os.path.join(paths.data, subj, f"{subj}_badchannels.tsv")
    path_event_dict = os.path.join(paths.data, subj, f"{subj}_event_dict.json")
    
    if not os.path.exists(path_task_data):
        print(f"  WARNING: Data file not found for {subj}")
        return None
    
    raw = mne.io.read_raw_fif(path_task_data, preload=True, verbose=False)
    
    if os.path.exists(path_bad_channels):
        bad_channels = pd.read_csv(path_bad_channels, sep='\t')
        bad_channels = bad_channels['badchannelslots'].tolist()
        raw.info['bads'].clear()
        raw.info['bads'].extend(bad_channels)
        print(f"  Bad channels: {bad_channels}")
    
    events = mne.find_events(raw, stim_channel='di32', verbose=False)
    
    with open(path_event_dict, 'r') as f:
        event_dict = json.load(f)
    
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-0.1, tmax=0.6, baseline=None, detrend=1,
        reject_by_annotation=False, preload=True, verbose=False
    )
    
    return epochs


def decode_and_extract_patterns(epochs, peak_time=None):
    """
    Run temporal decoding and extract patterns at peak.
    
    Returns
    -------
    scores : ndarray
        Cross-validated AUC scores (n_folds, n_times)
    times : ndarray
        Time points
    patterns : ndarray
        Topographical patterns at peak time
    info : mne.Info
        Channel info for plotting
    """
    
    epochs_freq = epochs["freq/tone"]
    epochs_infreq = epochs["infreq/tone"]
    
    print(f"  Frequent: {len(epochs_freq)} epochs, Infrequent: {len(epochs_infreq)} epochs")
    
    min_len = min(len(epochs_freq), len(epochs_infreq))
    
    if min_len < 20:
        print(f"  WARNING: Too few epochs ({min_len})")
        return None, None, None, None
    
    np.random.seed(42)
    idx_freq = np.random.choice(len(epochs_freq), min_len, replace=False)
    idx_infreq = np.random.choice(len(epochs_infreq), min_len, replace=False)
    
    epochs_equalized = mne.concatenate_epochs([
        epochs_freq[idx_freq], epochs_infreq[idx_infreq]
    ])
    
    print(f"  Equalized to {min_len} epochs per condition")
    
    # Get data - explicitly exclude bad channels
    epochs_mag = epochs_equalized.copy().pick('mag', exclude='bads')
    
    # Print channel info
    n_channels = len(epochs_mag.ch_names)
    n_bads = len(epochs_equalized.info['bads'])
    print(f"  Using {n_channels} channels (excluded {n_bads} bad)")
    
    X = epochs_mag.get_data(copy=True)
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
    times = epochs_mag.times
    
    # --- Temporal Decoding ---
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear', random_state=42)
    )
    
    time_decoder = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)
    
    print(f"  Mean AUC: {np.mean(scores):.3f}, Peak AUC: {np.max(np.mean(scores, axis=0)):.3f}")
    
    # --- Extract Patterns at Peak ---
    if peak_time is None:
        # Find peak from this subject's data
        mean_scores = np.mean(scores, axis=0)
        post_stim_mask = times >= 0
        peak_idx = np.argmax(mean_scores * post_stim_mask)
        peak_time = times[peak_idx]
    
    peak_idx = np.argmin(np.abs(times - peak_time))
    
    # Extract data at peak
    X_peak = X[:, :, peak_idx]
    
    # Fit LinearModel
    clf_pattern = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear', random_state=42))
    )
    clf_pattern.fit(X_peak, y)
    
    patterns = get_coef(clf_pattern, 'patterns_', inverse_transform=True)
    info = epochs_mag.info.copy()
    
    return scores, times, patterns, info


def identify_peak(mean_scores, times, min_time=0.0, prominence=0.01, distance=15):
    """Identify the largest peak in the mean decoding curve."""
    post_stim_mask = times >= min_time
    post_stim_scores = mean_scores.copy()
    post_stim_scores[~post_stim_mask] = 0
    
    peak_indices, _ = find_peaks(post_stim_scores, prominence=prominence, distance=distance)
    
    if len(peak_indices) > 1:
        largest_idx = np.argmax(mean_scores[peak_indices])
        peak_indices = np.array([peak_indices[largest_idx]])
    
    return peak_indices[0] if len(peak_indices) > 0 else np.argmax(post_stim_scores)


# =============================================================================
# PART 1: PROCESS ALL PARTICIPANTS
# =============================================================================

print("\n" + "="*70)
print("PART 1: PROCESSING ALL PARTICIPANTS")
print("="*70)

all_scores = {}
all_patterns = {}
all_info = {}
all_times = None
successful_subjects = []

for subj in subjects:
    print(f"\nProcessing {subj}...")
    
    try:
        epochs = load_epochs_for_subject(subj, paths)
        
        if epochs is None:
            continue
        
        scores, times, patterns, info = decode_and_extract_patterns(epochs)
        
        if scores is None:
            continue
        
        all_scores[subj] = np.mean(scores, axis=0)  # Average across CV folds
        all_patterns[subj] = patterns
        all_info[subj] = info
        successful_subjects.append(subj)
        
        if all_times is None:
            all_times = times
        
        print(f"  ✓ Complete")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"Successfully processed {len(successful_subjects)}/{len(subjects)} subjects")
print('='*60)

if len(successful_subjects) == 0:
    print("No subjects successfully processed!")
    exit()

# =============================================================================
# PART 2: COMPUTE GROUP STATISTICS
# =============================================================================

print("\n" + "="*70)
print("PART 2: COMPUTING GROUP STATISTICS")
print("="*70)

times = all_times
scores_array = np.array([all_scores[s] for s in successful_subjects])

# Group statistics
mean_scores_group = np.mean(scores_array, axis=0)
sem_scores_group = np.std(scores_array, axis=0) / np.sqrt(len(successful_subjects))

# Find peak
peak_idx = identify_peak(mean_scores_group, times)
peak_time = times[peak_idx]
peak_value = mean_scores_group[peak_idx]
scores_at_peak = scores_array[:, peak_idx]

print(f"Peak: {peak_time*1000:.1f} ms, AUC = {peak_value:.3f}")

# Binomial test
n_subjects = len(successful_subjects)
n_above_chance = np.sum(scores_at_peak > 0.5)
binomial_result = binomtest(n_above_chance, n_subjects, p=0.5, alternative='greater')

print(f"Binomial test: {n_above_chance}/{n_subjects} above chance, p = {binomial_result.pvalue:.4f}")

# Print MVPA results summary for paper
print("\n" + "-"*60)
print("MVPA RESULTS SUMMARY (for paper)")
print("-"*60)
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

Analysis: Temporal decoding of frequent vs infrequent tones
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

mvpa_summary_path = os.path.join(path_results, 'mvpa_results_summary.txt')
with open(mvpa_summary_path, 'w') as f:
    f.write(mvpa_summary)
print(f"\nSaved MVPA summary: {mvpa_summary_path}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'time_ms': times * 1000,
    'mean_auc': mean_scores_group,
    'sem_auc': sem_scores_group
})
for i, subj in enumerate(successful_subjects):
    results_df[subj] = scores_array[i]

results_csv_path = os.path.join(path_results, 'decoding_timecourse.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Saved timecourse data: {results_csv_path}")

# Save peak summary
peak_df = pd.DataFrame({
    'subject': successful_subjects,
    'auc_at_peak': scores_at_peak,
    'above_chance': scores_at_peak > 0.5
})
peak_csv_path = os.path.join(path_results, 'peak_subject_scores.csv')
peak_df.to_csv(peak_csv_path, index=False)
print(f"Saved peak scores: {peak_csv_path}")

# --- Re-extract patterns at GROUP peak time ---
print(f"\nRe-extracting patterns at group peak ({peak_time*1000:.1f} ms)...")

for subj in successful_subjects:
    epochs = load_epochs_for_subject(subj, paths)
    _, _, patterns, info = decode_and_extract_patterns(epochs, peak_time=peak_time)
    all_patterns[subj] = patterns
    all_info[subj] = info

# =============================================================================
# PART 3: RANK SUBJECTS BY AUC AT PEAK
# =============================================================================

# Create ranking
subject_aucs = {subj: scores_at_peak[i] for i, subj in enumerate(successful_subjects)}
ranked_subjects = sorted(subject_aucs.keys(), key=lambda x: subject_aucs[x], reverse=True)

# Assign colors based on original order (for consistency with legend)
subject_color_map = {subj: SUBJECT_COLORS[i % len(SUBJECT_COLORS)] 
                     for i, subj in enumerate(successful_subjects)}

print("\nSubject ranking by AUC at peak:")
for rank, subj in enumerate(ranked_subjects):
    print(f"  {rank+1}. {subj}: {subject_aucs[subj]:.3f}")

# =============================================================================
# PART 4: CREATE 3-PANEL FIGURE
# =============================================================================

print("\n" + "="*70)
print("PART 4: CREATING 3-PANEL FIGURE")
print("="*70)

# Apply smoothing for visualization
if SMOOTHING_SIGMA > 0:
    mean_scores_smooth = gaussian_filter1d(mean_scores_group, sigma=SMOOTHING_SIGMA)
    sem_scores_smooth = gaussian_filter1d(sem_scores_group, sigma=SMOOTHING_SIGMA)
    scores_array_smooth = np.array([gaussian_filter1d(s, sigma=SMOOTHING_SIGMA) for s in scores_array])
    
    # Recalculate peak on smoothed data
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
# Title will be added via fig.text for alignment with right panel
ax_temporal.legend(loc='upper right', fontsize=18, ncol=2)
ax_temporal.set_xlim([times[0]*1000, times[-1]*1000])
ax_temporal.set_ylim([0.35, 0.80])
ax_temporal.tick_params(axis='both', labelsize=20)

# Add title for temporal panel using fig.text for alignment
gs_pos_temporal = gs[0].get_position(fig)
fig.text((gs_pos_temporal.x0 + gs_pos_temporal.x1) / 2, 0.86, 
         'Temporal Decoding of\nFrequent vs Infrequent Tones', 
         fontsize=26, fontweight='bold', ha='center', va='bottom')

# --- RIGHT PANEL: Topographies (3 rows x 2 columns grid with colorbars) ---
# Order subjects by ID number (not by ranking)
ordered_subjects = sorted(successful_subjects)  # Alphabetical/numerical order

# Create nested GridSpec for topographies with tighter spacing
n_topo_rows = 3
n_topo_cols = 2
gs_topo = gridspec.GridSpecFromSubplotSpec(
    n_topo_rows, n_topo_cols, subplot_spec=gs[1], 
    wspace=0.6, hspace=0.55,
)

from mne.viz import plot_topomap

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

# Note: tight_layout called before frame drawing
# plt.tight_layout() - skip this as we're manually positioning

# Adjust subplot positions to make room for titles
fig.subplots_adjust(top=0.80)

# Save figure
if SAVE_FIGURES:
    fig_path = os.path.join(path_results, 'combined_decoding_figure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {fig_path}")
    
    fig_pdf = os.path.join(path_results, 'combined_decoding_figure.pdf')
    fig.savefig(fig_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_pdf}")

plt.show()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)