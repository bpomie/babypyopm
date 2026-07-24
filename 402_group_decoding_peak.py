#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 2025

Group-Level Temporal Decoding: Frequent vs Infrequent Tones
WITH Peak Statistical Analysis

This script:
1. Performs temporal decoding across multiple participants
2. Creates grand average decoding performance curves
3. Identifies the two main peaks in the mean decoding curve
4. Runs binomial tests at each peak to assess significance

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from scipy.signal import find_peaks
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

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

# Create results directories
path_results_decoding = os.path.join(
    root_data_path, 'results', 'preprocessing_routine_2', 'decoding_group'
)
os.makedirs(path_results_decoding, exist_ok=True)

path_results_peak_stats = os.path.join(
    root_data_path, 'results', 'decoding_peak_stats'
)
os.makedirs(path_results_peak_stats, exist_ok=True)

# List all subjects
subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")

# =============================================================================
# HELPER FUNCTIONS
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
        paths.data, subj, input_folder, f"{subj}_file-oddballTones_processed_2_filter_ica.fif"
    )
    path_bad_channels = os.path.join(paths.data, subj, f"{subj}_badchannels.tsv")
    path_event_dict = os.path.join(paths.data, subj, f"{subj}_event_dict.json")
    
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
        
        # Create epochs - No baseline correction for decoding
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=-0.1,
            tmax=0.5,
            baseline=None,
            detrend=1,
            reject_by_annotation=False,
            preload=True,
            verbose=False
        )
        
        # Select conditions
        epochs_freq = epochs["freq/tone"]
        epochs_infreq = epochs["infreq/tone"]
        
        print(f"Frequent: {len(epochs_freq)} epochs")
        print(f"Infrequent: {len(epochs_infreq)} epochs")
        
        # Equalize epoch counts
        min_len = min(len(epochs_freq), len(epochs_infreq))
        
        if min_len < 20:  # Minimum threshold
            print(f"WARNING: Too few epochs ({min_len}) for {subj}")
            return None, None
        
        # Random sampling to equalize
        np.random.seed(42)
        idx_freq = np.random.choice(len(epochs_freq), min_len, replace=False)
        idx_infreq = np.random.choice(len(epochs_infreq), min_len, replace=False)
        
        epochs_equalized = mne.concatenate_epochs([
            epochs_freq[idx_freq],
            epochs_infreq[idx_infreq]
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


def identify_peaks(mean_scores, times, min_time=0.0, prominence=0.01, distance=20):
    """
    Identify the largest peak in the mean decoding curve.
    
    Parameters
    ----------
    mean_scores : array
        Mean AUC scores across subjects
    times : array
        Time points
    min_time : float
        Only look for peaks after this time (default: 0, stimulus onset)
    prominence : float
        Minimum prominence for peak detection
    distance : int
        Minimum distance between peaks in samples
        
    Returns
    -------
    peak_indices : array
        Index of the largest peak
    peak_times : array
        Time of the largest peak
    peak_values : array
        AUC value at the peak
    """
    # Only consider post-stimulus period
    post_stim_mask = times >= min_time
    post_stim_scores = mean_scores.copy()
    post_stim_scores[~post_stim_mask] = 0  # Zero out pre-stimulus
    
    # Find peaks
    peak_indices, properties = find_peaks(
        post_stim_scores, 
        prominence=prominence,
        distance=distance
    )
    
    # Take only the largest peak
    if len(peak_indices) > 1:
        largest_idx = np.argmax(mean_scores[peak_indices])
        peak_indices = np.array([peak_indices[largest_idx]])
    
    peak_times = times[peak_indices]
    peak_values = mean_scores[peak_indices]
    
    return peak_indices, peak_times, peak_values


def run_binomial_test(scores_at_peak, chance_level=0.5):
    """
    Run binomial test to check if significantly more subjects
    are above chance than expected.
    
    Parameters
    ----------
    scores_at_peak : array
        AUC scores for each subject at a given time point
    chance_level : float
        The null hypothesis probability (default: 0.5)
        
    Returns
    -------
    dict with test results
    """
    n_subjects = len(scores_at_peak)
    n_above_chance = np.sum(scores_at_peak > chance_level)
    
    # Under null: each subject has 50% chance of being above 0.5
    result = binomtest(n_above_chance, n_subjects, p=0.5, alternative='greater')
    
    return {
        'n_subjects': n_subjects,
        'n_above_chance': n_above_chance,
        'proportion_above': n_above_chance / n_subjects,
        'expected_by_chance': n_subjects * 0.5,
        'p_value': result.pvalue,
        'statistic': n_above_chance,
        'mean_auc': np.mean(scores_at_peak),
        'std_auc': np.std(scores_at_peak),
        'individual_scores': scores_at_peak
    }


# =============================================================================
# PART 1: PROCESS ALL PARTICIPANTS
# =============================================================================

print("="*70)
print("PART 1: GROUP TEMPORAL DECODING")
print("="*70)

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
# PLOT GROUP RESULTS
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ----- Plot 1: Individual subjects + grand average -----
ax1.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance', zorder=1)
ax1.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

for i, subj in enumerate(successful_subjects):
    ax1.plot(times, scores_array[i], alpha=0.3, linewidth=1, color='gray', zorder=2)

ax1.plot(times, mean_scores_group, linewidth=2.5, color='blue',
         label=f'Grand Average (n={len(successful_subjects)})', zorder=3)
ax1.fill_between(times, mean_scores_group - sem_scores_group,
                 mean_scores_group + sem_scores_group, alpha=0.3, color='blue', zorder=2)

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.set_title('Group Temporal Decoding: Frequent vs Infrequent Tones\n'
              f'Individual Subjects (gray) and Grand Average (blue)', fontsize=13)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.4, 0.7])

# ----- Plot 2: Grand average with confidence interval -----
ax2.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance')
ax2.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

ax2.plot(times, mean_scores_group, linewidth=2.5, color='blue', label='Mean AUC')
ax2.fill_between(times, mean_scores_group - sem_scores_group,
                 mean_scores_group + sem_scores_group, alpha=0.3, color='blue', label='±SEM')

threshold = 0.55
sig_mask = mean_scores_group > threshold
if np.any(sig_mask):
    sig_indices = np.where(sig_mask)[0]
    if len(sig_indices) > 0:
        ax2.axvspan(times[sig_indices[0]], times[sig_indices[-1]],
                    alpha=0.15, color='green', label=f'AUC > {threshold}')

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('AUC Score', fontsize=12)
ax2.set_title(f'Grand Average Temporal Decoding (n={len(successful_subjects)} subjects)', fontsize=13)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.4, 0.7])

plt.tight_layout()

fig_path = os.path.join(path_results_decoding, 'group_temporal_decoding.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved group figure to: {fig_path}")
plt.show()

# =============================================================================
# GROUP DECODING SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print("GROUP DECODING SUMMARY")
print('='*60)

peak_idx = np.argmax(mean_scores_group)
peak_time = times[peak_idx]
peak_score = mean_scores_group[peak_idx]

print(f"\nPeak AUC: {peak_score:.3f} at {peak_time:.3f} s")
print(f"SEM at peak: {sem_scores_group[peak_idx]:.3f}")

post_stim_mask = times >= 0
mean_post_stim = mean_scores_group[post_stim_mask].mean()
print(f"\nMean AUC (0-0.5s): {mean_post_stim:.3f}")

baseline_mask = times < 0
mean_baseline = mean_scores_group[baseline_mask].mean()
print(f"Mean AUC (baseline): {mean_baseline:.3f}")

above_threshold = np.where(mean_scores_group > threshold)[0]
if len(above_threshold) > 0:
    first_sig_idx = above_threshold[0]
    last_sig_idx = above_threshold[-1]
    print(f"\nFirst time above {threshold}: {times[first_sig_idx]:.3f} s")
    print(f"Last time above {threshold}: {times[last_sig_idx]:.3f} s")
    print(f"Duration above threshold: {times[last_sig_idx] - times[first_sig_idx]:.3f} s")

print(f"\nSubject-wise peak AUC:")
subject_peaks = np.max(scores_array, axis=1)
print(f"  Mean: {np.mean(subject_peaks):.3f}")
print(f"  Std: {np.std(subject_peaks):.3f}")
print(f"  Range: [{np.min(subject_peaks):.3f}, {np.max(subject_peaks):.3f}]")

# =============================================================================
# SAVE GROUP RESULTS
# =============================================================================

results_dict = {
    'times': times,
    'scores_array': scores_array,
    'mean_scores_group': mean_scores_group,
    'sem_scores_group': sem_scores_group,
    'subjects': successful_subjects,
    'n_subjects': len(successful_subjects)
}

np.save(os.path.join(path_results_decoding, 'group_decoding_results.npy'),
        results_dict, allow_pickle=True)

results_df = pd.DataFrame({
    'time': times,
    'mean_auc': mean_scores_group,
    'sem_auc': sem_scores_group
})

for i, subj in enumerate(successful_subjects):
    results_df[subj] = scores_array[i]

results_df.to_csv(os.path.join(path_results_decoding, 'group_decoding_results.csv'), index=False)
print(f"\nSaved results to: {path_results_decoding}")

# =============================================================================
# PART 2: PEAK STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("PART 2: PEAK STATISTICAL ANALYSIS")
print("="*70)

# =============================================================================
# IDENTIFY PEAKS
# =============================================================================

print("\n" + "-"*50)
print("PEAK IDENTIFICATION")
print("-"*50)

peak_indices, peak_times_detected, peak_values = identify_peaks(
    mean_scores_group, times, 
    min_time=0.0, 
    prominence=0.01,
    distance=15
)

print(f"\nFound {len(peak_indices)} peak:")
for i, (idx, t, v) in enumerate(zip(peak_indices, peak_times_detected, peak_values)):
    print(f"  Peak: time = {t*1000:.1f} ms, AUC = {v:.3f}")

# =============================================================================
# STATISTICAL TESTS AT EACH PEAK
# =============================================================================

print("\n" + "-"*50)
print("BINOMIAL TESTS AT PEAKS")
print("-"*50)

print("\nNull hypothesis: Each subject has 50% probability of AUC > 0.5")
print("Alternative: More subjects are above 0.5 than expected by chance\n")

peak_results = []

for i, (p_idx, p_time, p_value) in enumerate(zip(peak_indices, peak_times_detected, peak_values)):
    print(f"\n--- Peak (t = {p_time*1000:.1f} ms) ---")
    
    scores_at_peak = scores_array[:, p_idx]
    result = run_binomial_test(scores_at_peak)
    result['peak_time'] = p_time
    result['peak_idx'] = p_idx
    peak_results.append(result)
    
    print(f"  Mean AUC at peak: {result['mean_auc']:.3f} (SD = {result['std_auc']:.3f})")
    print(f"  Subjects above 0.5: {result['n_above_chance']}/{result['n_subjects']}")
    print(f"  Expected by chance: {result['expected_by_chance']:.1f}")
    print(f"  Binomial test p-value: {result['p_value']:.4f}")
    
    if result['p_value'] < 0.05:
        print(f"  --> SIGNIFICANT at α = 0.05")
    else:
        print(f"  --> Not significant at α = 0.05")
    
    print(f"\n  Individual scores at peak:")
    for j, (subj, score) in enumerate(zip(successful_subjects, scores_at_peak)):
        marker = "*" if score > 0.5 else " "
        print(f"    {subj}: {score:.3f} {marker}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "-"*50)
print("SUMMARY TABLE")
print("-"*50)

summary_df = pd.DataFrame({
    'Peak': ['Peak' for i in range(len(peak_results))],
    'Time (ms)': [r['peak_time']*1000 for r in peak_results],
    'Mean AUC': [r['mean_auc'] for r in peak_results],
    'N above 0.5': [r['n_above_chance'] for r in peak_results],
    'N total': [r['n_subjects'] for r in peak_results],
    'p-value': [r['p_value'] for r in peak_results],
    'Significant': ['Yes' if r['p_value'] < 0.05 else 'No' for r in peak_results]
})

print("\n", summary_df.to_string(index=False))

# =============================================================================
# PEAK STATISTICS VISUALIZATION
# =============================================================================

print("\n" + "-"*50)
print("Creating peak statistics visualization...")
print("-"*50)

fig2, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Plot 1: Group decoding with peaks marked ---
ax1 = axes[0]

ax1.axhline(0.5, color='k', linestyle='--', linewidth=1.5, label='Chance (0.5)')
ax1.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

for i, subj in enumerate(successful_subjects):
    ax1.plot(times * 1000, scores_array[i], alpha=0.3, linewidth=1, color='gray')

ax1.plot(times * 1000, mean_scores_group, linewidth=2.5, color='blue', label='Grand Average')
ax1.fill_between(times * 1000, mean_scores_group - sem_scores_group,
                 mean_scores_group + sem_scores_group, alpha=0.3, color='blue')

for i, (p_idx, p_time, p_value) in enumerate(zip(peak_indices, peak_times_detected, peak_values)):
    color = 'green' if peak_results[i]['p_value'] < 0.05 else 'orange'
    ax1.scatter(p_time * 1000, p_value, s=150, color=color, 
                zorder=5, edgecolor='black', linewidth=2)
    ax1.annotate(
        f'Peak\n{p_time*1000:.0f}ms\np={peak_results[i]["p_value"]:.3f}',
        xy=(p_time * 1000, p_value),
        xytext=(p_time * 1000 + 30, p_value + 0.03),
        fontsize=10, ha='left',
        arrowprops=dict(arrowstyle='->', color='black', lw=1)
    )

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.set_title('Temporal Decoding with Identified Peaks', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([times[0]*1000, times[-1]*1000])
ax1.set_ylim([0.4, 0.75])

# --- Plot 2: Subject-level scores at peaks ---
ax2 = axes[1]

n_subjects = len(successful_subjects)
x_positions = np.arange(len(peak_results))
width = 0.12
colors = plt.cm.tab10(np.linspace(0, 1, n_subjects))

for j, subj in enumerate(successful_subjects):
    subject_scores = [r['individual_scores'][j] for r in peak_results]
    ax2.bar(x_positions + j*width - width*n_subjects/2 + width/2, 
            subject_scores, width, label=subj, color=colors[j], alpha=0.8)

ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance')

ax2.set_xlabel('Peak', fontsize=12)
ax2.set_ylabel('AUC Score', fontsize=12)
ax2.set_title('Individual Subject Scores at Each Peak', fontsize=14, fontweight='bold')
ax2.set_xticks(x_positions)
ax2.set_xticklabels([f'Peak\n({r["peak_time"]*1000:.0f}ms)' for i, r in enumerate(peak_results)])
ax2.legend(loc='upper right', ncol=3, fontsize=9)
ax2.set_ylim([0.4, 0.75])
ax2.grid(True, alpha=0.3, axis='y')

for i, r in enumerate(peak_results):
    sig_marker = '*' if r['p_value'] < 0.05 else ''
    ax2.text(i, 0.72, f"n>0.5: {r['n_above_chance']}/{r['n_subjects']}\np={r['p_value']:.3f}{sig_marker}", 
             ha='center', fontsize=10)

plt.tight_layout()

output_path = os.path.join(path_results_peak_stats, 'decoding_peak_statistics.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
plt.show()

# =============================================================================
# SAVE PEAK STATISTICS RESULTS
# =============================================================================

summary_path = os.path.join(path_results_peak_stats, 'decoding_peak_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

detailed_data = []
for i, r in enumerate(peak_results):
    for j, (subj, score) in enumerate(zip(successful_subjects, r['individual_scores'])):
        detailed_data.append({
            'peak': 'Peak',
            'peak_time_ms': r['peak_time']*1000,
            'subject': subj,
            'auc': score,
            'above_chance': score > 0.5
        })

detailed_df = pd.DataFrame(detailed_data)
detailed_path = os.path.join(path_results_peak_stats, 'decoding_peak_detailed.csv')
detailed_df.to_csv(detailed_path, index=False)
print(f"Detailed results saved to: {detailed_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nGroup decoding results saved to:\n  {path_results_decoding}")
print(f"\nPeak statistics saved to:\n  {path_results_peak_stats}")
print("\n" + "="*70)