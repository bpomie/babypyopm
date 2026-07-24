#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
402b_group_decoding_figures.py

FIGURES ONLY - Loads pre-computed results and creates visualizations.
Run 402a_group_decoding_compute.py first to generate the data.

Iterate on this script to improve figures without re-running decoding.

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binom
import os

# =============================================================================
# PARAMETERS
# =============================================================================

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

path_results_peak_stats = os.path.join(root_data_path, 'results', 'decoding_peak_stats')

# Figure parameters - adjust these to iterate on visualization
SMOOTHING_SIGMA = 10  # Gaussian smoothing sigma (in samples). Set to 0 for no smoothing.
FIGURE_DPI = 300
SAVE_FIGURES = True

# =============================================================================
# LOAD PRE-COMPUTED RESULTS
# =============================================================================

print("Loading pre-computed results...")

results_path = os.path.join(path_results_peak_stats, 'decoding_results_for_figures.npy')

if not os.path.exists(results_path):
    raise FileNotFoundError(
        f"Could not find: {results_path}\n"
        "Run 402a_group_decoding_compute.py first!"
    )

results = np.load(results_path, allow_pickle=True).item()

# Unpack results
times = results['times']
scores_array = results['scores_array']
mean_scores_group = results['mean_scores_group']
sem_scores_group = results['sem_scores_group']
successful_subjects = results['successful_subjects']
peak_idx = results['peak_idx']
peak_time = results['peak_time']
peak_value = results['peak_value']
scores_at_peak = results['scores_at_peak']
binomial_result = results['binomial_result']

n_subjects = len(successful_subjects)

print(f"Loaded data for {n_subjects} subjects")
print(f"Peak at {peak_time*1000:.1f} ms, AUC = {peak_value:.3f}")
print(f"Binomial test: {binomial_result['n_above_chance']}/{n_subjects} above chance, p = {binomial_result['p_value']:.4f}")

# =============================================================================
# APPLY GAUSSIAN SMOOTHING (for visualization only)
# =============================================================================

if SMOOTHING_SIGMA > 0:
    print(f"\nApplying Gaussian smoothing (sigma={SMOOTHING_SIGMA})...")
    mean_scores_smooth = gaussian_filter1d(mean_scores_group, sigma=SMOOTHING_SIGMA)
    sem_scores_smooth = gaussian_filter1d(sem_scores_group, sigma=SMOOTHING_SIGMA)
    scores_array_smooth = np.array([gaussian_filter1d(s, sigma=SMOOTHING_SIGMA) for s in scores_array])
    
    # Recalculate peak on smoothed data for visualization
    post_stim_mask = times >= 0
    peak_idx_smooth = np.argmax(mean_scores_smooth * post_stim_mask)
    peak_time_smooth = times[peak_idx_smooth]
    peak_value_smooth = mean_scores_smooth[peak_idx_smooth]
else:
    mean_scores_smooth = mean_scores_group
    sem_scores_smooth = sem_scores_group
    scores_array_smooth = scores_array
    peak_idx_smooth = peak_idx
    peak_time_smooth = peak_time
    peak_value_smooth = peak_value

# =============================================================================
# FIGURE: LINE PLOT
# =============================================================================

print("\nCreating figure...")

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# --- Color scheme ---
# Using a perceptually distinct palette good for 6-8 subjects
subject_colors = [
    '#E64B35',  # red
    '#4DBBD5',  # cyan
    '#00A087',  # teal
    '#3C5488',  # navy
    '#F39B7F',  # salmon
    '#8491B4',  # slate
    '#91D1C2',  # mint
    '#DC9E82',  # tan
]
# Trim to number of subjects
subject_colors = subject_colors[:n_subjects]

# Reference lines
ax1.axhline(0.5, color='k', linestyle='--', linewidth=1.5, label='Chance', zorder=1)
ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

# Individual subjects (smoothed) with colors
for i, subj in enumerate(successful_subjects):
    label = subj.replace('sub-', 'S')  # Shorter labels: S101, S102, etc.
    ax1.plot(times * 1000, scores_array_smooth[i], 
             alpha=0.2, linewidth=1.5, color=subject_colors[i], label=label)

# Grand average with SEM (smoothed)
ax1.plot(times * 1000, mean_scores_smooth, linewidth=3, color='black', 
         label='Grand Average', zorder=4)
ax1.fill_between(
    times * 1000,
    mean_scores_smooth - sem_scores_smooth,
    mean_scores_smooth + sem_scores_smooth,
    alpha=0.2, color='black', zorder=3
)

# Annotation - positioned vertically below peak
sig_text = '*' if binomial_result['p_value'] < 0.05 else ''
ax1.annotate(
    f'{peak_time_smooth*1000:.0f} ms\np={binomial_result["p_value"]:.3f}{sig_text}',
    xy=(peak_time_smooth * 1000, peak_value_smooth),
    xytext=(peak_time_smooth * 1000, 0.42),
    fontsize=12, ha='center',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

ax1.set_xlabel('Time (ms)', fontsize=14)
ax1.set_ylabel('AUC', fontsize=14)
ax1.set_title('Temporal Decoding: Frequent vs Infrequent', fontsize=15, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10, ncol=2)
ax1.set_xlim([times[0]*1000, times[-1]*1000])
ax1.set_ylim([0.4, 0.75])
ax1.tick_params(axis='both', labelsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    output_path = os.path.join(path_results_peak_stats, 'decoding_temporal.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

plt.show()

# =============================================================================
# FIGURE 2: BINOMIAL DISTRIBUTION (optional)
# =============================================================================

fig2, ax = plt.subplots(figsize=(8, 5))

# Binomial distribution under null
k_values = np.arange(0, n_subjects + 1)
probabilities = binom.pmf(k_values, n_subjects, 0.5)

# All bars
bars = ax.bar(k_values, probabilities, color='steelblue', alpha=0.7, edgecolor='black')

# Highlight tail (observed and more extreme)
observed_k = binomial_result['n_above_chance']
for i, k in enumerate(k_values):
    if k >= observed_k:
        bars[i].set_color('tomato')
        bars[i].set_alpha(1.0)

# Reference lines
ax.axvline(observed_k, color='red', linestyle='--', linewidth=2, label=f'Observed: {observed_k}/{n_subjects}')
ax.axvline(n_subjects * 0.5, color='gray', linestyle=':', linewidth=2, label=f'Expected: {n_subjects * 0.5:.1f}')

# Labels
ax.set_xlabel('Number of subjects above chance', fontsize=12)
ax.set_ylabel('Probability under null', fontsize=12)
ax.set_title(f'Binomial Test (n={n_subjects}, p=0.5)', fontsize=13, fontweight='bold')
ax.set_xticks(k_values)
ax.set_xticklabels([f'{k}' for k in k_values])

# Probability labels on bars
for k, prob in zip(k_values, probabilities):
    ax.text(k, prob + 0.005, f'{prob:.3f}', ha='center', fontsize=9)

# P-value annotation
ax.annotate(
    f'p = {binomial_result["p_value"]:.4f}\n(shaded area)',
    xy=(observed_k, probabilities[observed_k]),
    xytext=(observed_k - 1.5, probabilities[observed_k] + 0.12),
    fontsize=11, ha='center',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

ax.legend(loc='upper left', fontsize=10)
ax.set_ylim([0, 0.4])

plt.tight_layout()

if SAVE_FIGURES:
    output_path2 = os.path.join(path_results_peak_stats, 'binomial_distribution.png')
    plt.savefig(output_path2, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {output_path2}")

plt.show()

print("\n" + "="*70)
print("FIGURES COMPLETE")
print("="*70)