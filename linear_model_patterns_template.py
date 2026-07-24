#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
402c_group_decoding_patterns.py

Extract topographical patterns from LinearModel at peak decoding time.
Based on: https://mne.tools/stable/auto_examples/decoding/linear_model_patterns.html

Run AFTER 402a_group_decoding_compute.py (uses saved peak time).

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all

import mne
from mne.decoding import LinearModel, get_coef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_2_filter_ica'
task = 'oddballTones'

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# Peak time - will be loaded from previous results, or set manually
PEAK_TIME = 0.233  # seconds (fallback if results not found)

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

path_results_peak_stats = os.path.join(
    root_data_path, 'results', 'decoding_peak_stats'
)

path_results_patterns = os.path.join(
    root_data_path, 'results', 'decoding_patterns'
)
os.makedirs(path_results_patterns, exist_ok=True)

# Try to load peak time from previous results
try:
    prev_results = np.load(
        os.path.join(path_results_peak_stats, 'decoding_results_for_figures.npy'),
        allow_pickle=True
    ).item()
    PEAK_TIME = prev_results['peak_time']
    print(f"Loaded peak time from previous results: {PEAK_TIME*1000:.1f} ms")
except:
    print(f"Using default peak time: {PEAK_TIME*1000:.1f} ms")

subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_epochs_for_decoding(subj, paths):
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
        tmin=-0.1, tmax=0.5, baseline=None, detrend=1,
        reject_by_annotation=False, preload=True, verbose=False
    )
    
    return epochs


def fit_linear_model_at_peak(epochs, peak_time):
    """
    Fit LinearModel at peak time and extract patterns.
    
    Returns
    -------
    patterns : ndarray
        Topographical patterns (n_channels,)
    info : mne.Info
        Channel info for plotting
    """
    
    # Get epochs for each condition
    epochs_freq = epochs["freq/tone"]
    epochs_infreq = epochs["infreq/tone"]
    
    print(f"  Frequent: {len(epochs_freq)} epochs")
    print(f"  Infrequent: {len(epochs_infreq)} epochs")
    
    # Balance classes
    min_len = min(len(epochs_freq), len(epochs_infreq))
    
    if min_len < 10:
        print(f"  WARNING: Too few epochs ({min_len})")
        return None, None
    
    np.random.seed(42)
    idx_freq = np.random.choice(len(epochs_freq), min_len, replace=False)
    idx_infreq = np.random.choice(len(epochs_infreq), min_len, replace=False)
    
    epochs_equalized = mne.concatenate_epochs([
        epochs_freq[idx_freq], epochs_infreq[idx_infreq]
    ])
    
    print(f"  Equalized to {min_len} epochs per condition")
    
    # Pick mag channels
    epochs_mag = epochs_equalized.copy().pick('mag')
    
    # Get data at peak time
    times = epochs_mag.times
    peak_idx = np.argmin(np.abs(times - peak_time))
    actual_peak = times[peak_idx]
    print(f"  Extracting at {actual_peak*1000:.1f} ms")
    
    # Extract data at peak: (n_epochs, n_channels)
    X = epochs_mag.get_data(copy=True)[:, :, peak_idx]
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
    
    # Build pipeline with LinearModel wrapper
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear', random_state=42))
    )
    
    # Fit the model
    clf.fit(X, y)
    
    # Extract patterns (neurophysiologically interpretable)
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)
    
    # Get info for plotting
    info = epochs_mag.info.copy()
    
    return patterns, info


def plot_pattern_topomap(patterns, info, ax, title="", colorbar=False, cbar_ax=None):
    """Plot a single topographical pattern."""
    
    evoked = mne.EvokedArray(patterns[:, np.newaxis], info, tmin=0)
    
    if colorbar and cbar_ax is not None:
        # Provide both axes: topomap axis and colorbar axis
        evoked.plot_topomap(
            times=0,
            axes=[ax, cbar_ax],
            show=False,
            colorbar=True,
            scalings=1,
            time_format="",
        )
    else:
        evoked.plot_topomap(
            times=0,
            axes=ax,
            show=False,
            colorbar=False,
            scalings=1,
            time_format="",
        )
    
    ax.set_title(title, fontsize=12, fontweight='bold')


# =============================================================================
# MAIN: EXTRACT PATTERNS FOR ALL PARTICIPANTS
# =============================================================================

print("\n" + "="*70)
print(f"EXTRACTING LINEAR MODEL PATTERNS AT {PEAK_TIME*1000:.0f} ms")
print("="*70)

all_patterns = {}
all_info = {}
successful_subjects = []

for subj in subjects:
    print(f"\nProcessing {subj}...")
    
    try:
        epochs = load_epochs_for_decoding(subj, paths)
        
        if epochs is None:
            continue
        
        patterns, info = fit_linear_model_at_peak(epochs, PEAK_TIME)
        
        if patterns is None:
            continue
        
        all_patterns[subj] = patterns
        all_info[subj] = info
        successful_subjects.append(subj)
        
        print(f"  ✓ Patterns extracted ({len(patterns)} channels)")
        
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
# FIGURE: INDIVIDUAL PARTICIPANT PATTERNS
# =============================================================================

print("\nCreating individual patterns figure...")

n_subjects = len(successful_subjects)
n_cols = min(3, n_subjects)
n_rows = int(np.ceil(n_subjects / n_cols))

# Option 1: Without individual colorbars (cleaner look)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

if n_subjects == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, subj in enumerate(successful_subjects):
    ax = axes[idx]
    
    # Use actual subject ID as label
    short_label = subj.replace('sub-', 'S')  # e.g., "sub-101" -> "S101"
    
    # Plot without colorbar
    evoked = mne.EvokedArray(all_patterns[subj][:, np.newaxis], all_info[subj], tmin=0)
    evoked.plot_topomap(
        times=0,
        axes=ax,
        show=False,
        colorbar=False,
        scalings=1,
        time_format="",
    )
    ax.set_title(short_label, fontsize=12, fontweight='bold')

# Hide unused axes
for idx in range(n_subjects, len(axes)):
    axes[idx].axis('off')

fig.suptitle(f"Decoding Patterns at {PEAK_TIME*1000:.0f} ms\n(Frequent vs Infrequent)", 
              fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
fig_path = os.path.join(path_results_patterns, f'individual_patterns_{PEAK_TIME*1000:.0f}ms.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig_path}")

fig_pdf = os.path.join(path_results_patterns, f'individual_patterns_{PEAK_TIME*1000:.0f}ms.pdf')
fig.savefig(fig_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig_pdf}")

# =============================================================================
# FIGURE 2: WITH INDIVIDUAL COLORBARS
# =============================================================================

print("\nCreating figure with individual colorbars...")

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

if n_subjects == 1:
    axes2 = np.array([axes2])
axes2 = axes2.flatten()

for idx, subj in enumerate(successful_subjects):
    ax = axes2[idx]
    
    # Use actual subject ID as label
    short_label = subj.replace('sub-', 'S')  # e.g., "sub-101" -> "S101"
    
    # Get pattern data
    pattern_data = all_patterns[subj]
    
    # Print range for debugging
    print(f"  {subj}: pattern range = [{pattern_data.min():.2e}, {pattern_data.max():.2e}]")
    
    # Create evoked - set channel type to 'misc' to avoid unit scaling issues
    info_copy = all_info[subj].copy()
    
    # Create evoked and plot using mne.viz.plot_topomap directly for more control
    from mne.viz import plot_topomap
    
    # Get sensor positions
    pos = mne.channels.layout._find_topomap_coords(info_copy, picks='mag')
    
    # Plot with explicit vlim
    vmax = np.abs(pattern_data).max()
    vmin = -vmax
    
    im, _ = plot_topomap(
        pattern_data, 
        pos, 
        axes=ax,
        show=False,
        vlim=(vmin, vmax),
        cmap='RdBu_r',
    )
    
    # Add colorbar with units label
    # Note: LinearModel patterns are in arbitrary units (a.u.)
    # They represent the activation pattern, not physical measurements
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('a.u.', fontsize=8)  # arbitrary units
    
    ax.set_title(short_label, fontsize=12, fontweight='bold')

# Hide unused axes
for idx in range(n_subjects, len(axes2)):
    axes2[idx].axis('off')

fig2.suptitle(f"Decoding Patterns at {PEAK_TIME*1000:.0f} ms\n(Frequent vs Infrequent)", 
              fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure with colorbars
fig2_path = os.path.join(path_results_patterns, f'individual_patterns_{PEAK_TIME*1000:.0f}ms_with_colorbars.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig2_path}")

fig2_pdf = os.path.join(path_results_patterns, f'individual_patterns_{PEAK_TIME*1000:.0f}ms_with_colorbars.pdf')
fig2.savefig(fig2_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig2_pdf}")

plt.show()

print("\n" + "="*70)
print("PATTERN EXTRACTION COMPLETE")
print("="*70)