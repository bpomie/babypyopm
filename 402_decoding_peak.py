#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoding Peak Statistical Analysis

This script builds on 401_group_temporal_decoding to:
1. Identify the two main peaks in the mean decoding curve
2. Run binomial tests at each peak to assess whether significantly
   more subjects than expected by chance have AUC > 0.5

@author: b.pomiechowska@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.signal import find_peaks
import os
import pandas as pd

# =============================================================================
# PARAMETERS - Adjust these paths for your system
# =============================================================================

task = 'oddballTones'

# Insert the path to your project folder
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# Create results directory
path_results_decoding = os.path.join(
    root_data_path, 'results', 'decoding_peak_stats'
)
os.makedirs(path_results_decoding, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def identify_peaks(mean_scores, times, min_time=0.0, prominence=0.01, distance=20):
    """
    Identify peaks in the mean decoding curve.
    
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
        Indices of detected peaks
    peak_times : array
        Times of detected peaks
    peak_values : array
        AUC values at peaks
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
    
    # Sort by peak height and take top 2
    if len(peak_indices) > 2:
        sorted_idx = np.argsort(mean_scores[peak_indices])[::-1]
        peak_indices = peak_indices[sorted_idx[:2]]
        peak_indices = np.sort(peak_indices)  # Re-sort by time
    
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
    # (since AUC is symmetric around 0.5 under null)
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
# MAIN ANALYSIS
# =============================================================================

print("="*70)
print("DECODING PEAK STATISTICAL ANALYSIS")
print("="*70)

# =============================================================================
# LOAD DATA FROM GROUP DECODING RESULTS
# =============================================================================

path_decoding_csv = os.path.join(
    root_data_path, 'results', 'preprocessing_routine_2', 'decoding_group', 
    'group_decoding_results.csv'
)

if os.path.exists(path_decoding_csv):
    print(f"\nLoading decoding results from:\n  {path_decoding_csv}")
    results_df = pd.read_csv(path_decoding_csv)
    times = results_df['time'].values
    subject_cols = [c for c in results_df.columns if c.startswith('sub-')]
    scores_array = results_df[subject_cols].values.T  # Shape: (n_subjects, n_times)
    successful_subjects = subject_cols
else:
    raise FileNotFoundError(f"Could not find: {path_decoding_csv}")

n_subjects = len(successful_subjects)
print(f"\nNumber of subjects: {n_subjects}")
print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} s")
print(f"Number of time points: {len(times)}")

# Compute group statistics
mean_scores_group = np.mean(scores_array, axis=0)
sem_scores_group = np.std(scores_array, axis=0) / np.sqrt(n_subjects)

# =============================================================================
# IDENTIFY PEAKS
# =============================================================================

print("\n" + "="*70)
print("PEAK IDENTIFICATION")
print("="*70)

peak_indices, peak_times, peak_values = identify_peaks(
    mean_scores_group, times, 
    min_time=0.0, 
    prominence=0.01,
    distance=15
)

print(f"\nFound {len(peak_indices)} peaks:")
for i, (idx, t, v) in enumerate(zip(peak_indices, peak_times, peak_values)):
    print(f"  Peak {i+1}: time = {t*1000:.1f} ms, AUC = {v:.3f}")

# =============================================================================
# STATISTICAL TESTS AT EACH PEAK
# =============================================================================

print("\n" + "="*70)
print("BINOMIAL TESTS AT PEAKS")
print("="*70)

print("\nNull hypothesis: Each subject has 50% probability of AUC > 0.5")
print("Alternative: More subjects are above 0.5 than expected by chance\n")

peak_results = []

for i, (peak_idx, peak_time, peak_value) in enumerate(zip(peak_indices, peak_times, peak_values)):
    print(f"\n--- Peak {i+1} (t = {peak_time*1000:.1f} ms) ---")
    
    # Get scores at this peak for all subjects
    scores_at_peak = scores_array[:, peak_idx]
    
    # Run binomial test
    result = run_binomial_test(scores_at_peak)
    result['peak_time'] = peak_time
    result['peak_idx'] = peak_idx
    peak_results.append(result)
    
    print(f"  Mean AUC at peak: {result['mean_auc']:.3f} (SD = {result['std_auc']:.3f})")
    print(f"  Subjects above 0.5: {result['n_above_chance']}/{result['n_subjects']}")
    print(f"  Expected by chance: {result['expected_by_chance']:.1f}")
    print(f"  Binomial test p-value: {result['p_value']:.4f}")
    
    if result['p_value'] < 0.05:
        print(f"  --> SIGNIFICANT at α = 0.05")
    else:
        print(f"  --> Not significant at α = 0.05")
    
    # Show individual subject scores
    print(f"\n  Individual scores at peak:")
    for j, (subj, score) in enumerate(zip(successful_subjects, scores_at_peak)):
        marker = "*" if score > 0.5 else " "
        print(f"    {subj}: {score:.3f} {marker}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame({
    'Peak': [f'Peak {i+1}' for i in range(len(peak_results))],
    'Time (ms)': [r['peak_time']*1000 for r in peak_results],
    'Mean AUC': [r['mean_auc'] for r in peak_results],
    'N above 0.5': [r['n_above_chance'] for r in peak_results],
    'N total': [r['n_subjects'] for r in peak_results],
    'p-value': [r['p_value'] for r in peak_results],
    'Significant': ['Yes' if r['p_value'] < 0.05 else 'No' for r in peak_results]
})

print("\n", summary_df.to_string(index=False))

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Plot 1: Group decoding with peaks marked ---
ax1 = axes[0]

ax1.axhline(0.5, color='k', linestyle='--', linewidth=1.5, label='Chance (0.5)')
ax1.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Plot individual subjects
for i, subj in enumerate(successful_subjects):
    ax1.plot(times * 1000, scores_array[i], alpha=0.3, linewidth=1, color='gray')

# Plot mean with SEM
ax1.plot(times * 1000, mean_scores_group, linewidth=2.5, color='blue', label='Grand Average')
ax1.fill_between(
    times * 1000,
    mean_scores_group - sem_scores_group,
    mean_scores_group + sem_scores_group,
    alpha=0.3, color='blue'
)

# Mark peaks
for i, (peak_idx, peak_time, peak_value) in enumerate(zip(peak_indices, peak_times, peak_values)):
    color = 'green' if peak_results[i]['p_value'] < 0.05 else 'orange'
    ax1.scatter(peak_time * 1000, peak_value, s=150, color=color, 
                zorder=5, edgecolor='black', linewidth=2)
    ax1.annotate(
        f'Peak {i+1}\n{peak_time*1000:.0f}ms\np={peak_results[i]["p_value"]:.3f}',
        xy=(peak_time * 1000, peak_value),
        xytext=(peak_time * 1000 + 30, peak_value + 0.03),
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

x_positions = np.arange(len(peak_results))
width = 0.12
colors = plt.cm.tab10(np.linspace(0, 1, n_subjects))

# Plot individual subject bars
for j, subj in enumerate(successful_subjects):
    subject_scores = [r['individual_scores'][j] for r in peak_results]
    ax2.bar(x_positions + j*width - width*n_subjects/2 + width/2, 
            subject_scores, width, label=subj, color=colors[j], alpha=0.8)

# Add chance line
ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance')

# Formatting
ax2.set_xlabel('Peak', fontsize=12)
ax2.set_ylabel('AUC Score', fontsize=12)
ax2.set_title('Individual Subject Scores at Each Peak', fontsize=14, fontweight='bold')
ax2.set_xticks(x_positions)
ax2.set_xticklabels([f'Peak {i+1}\n({r["peak_time"]*1000:.0f}ms)' for i, r in enumerate(peak_results)])
ax2.legend(loc='upper right', ncol=3, fontsize=9)
ax2.set_ylim([0.4, 0.75])
ax2.grid(True, alpha=0.3, axis='y')

# Add annotations for binomial test results
for i, r in enumerate(peak_results):
    sig_marker = '*' if r['p_value'] < 0.05 else ''
    ax2.text(i, 0.72, f"n>{0.5}: {r['n_above_chance']}/{r['n_subjects']}\np={r['p_value']:.3f}{sig_marker}", 
             ha='center', fontsize=10)

plt.tight_layout()

# Save figure
output_path = os.path.join(path_results_decoding, 'decoding_peak_statistics.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

# Save summary
summary_path = os.path.join(path_results_decoding, 'decoding_peak_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# Save detailed results
detailed_data = []
for i, r in enumerate(peak_results):
    for j, (subj, score) in enumerate(zip(successful_subjects, r['individual_scores'])):
        detailed_data.append({
            'peak': i+1,
            'peak_time_ms': r['peak_time']*1000,
            'subject': subj,
            'auc': score,
            'above_chance': score > 0.5
        })

detailed_df = pd.DataFrame(detailed_data)
detailed_path = os.path.join(path_results_decoding, 'decoding_peak_detailed.csv')
detailed_df.to_csv(detailed_path, index=False)
print(f"Detailed results saved to: {detailed_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)