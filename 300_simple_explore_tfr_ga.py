#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Frequency Analysis Script - Group Analysis
Based on single participant TFR analysis

@author: b.pomiechowska@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt

import mne
import os
import pandas as pd
import json

def equalize_and_combine_epochs(epochs_list, seed=None):
    """Randomly equalize epoch counts and return a combined Epochs object."""
    if seed is not None:
        np.random.seed(seed)
    
    # Find minimum number of epochs
    min_len = min(len(epochs) for epochs in epochs_list)
    
    # Randomly downsample each condition
    equalized = []
    for epochs in epochs_list:
        indices = np.random.choice(len(epochs), min_len, replace=False)
        equalized.append(epochs[indices])
    
    # Combine into one Epochs object
    combined = mne.concatenate_epochs(equalized)
    return combined

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Insert the path to your project folder
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# PARTICIPANT LIST
# =============================================================================

# List all subjects in the data folder
path_data = os.path.join(root_data_path, 'data')
subjects = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f)) and f.startswith('sub-')]
print(f"Found subjects: {subjects}")

# Alternatively, manually specify subjects:
# subjects = ['sub-101', 'sub-102', 'sub-107']

# =============================================================================
# PATHS
# =============================================================================

# Results paths
path_results_tfr = os.path.join(root_data_path, 'results', 'tfr_analysis')
path_results_individual = os.path.join(path_results_tfr, 'individual')
path_results_group = os.path.join(path_results_tfr, 'group_average')

# Create directories
os.makedirs(path_results_individual, exist_ok=True)
os.makedirs(path_results_group, exist_ok=True)

print(f"Individual results: {path_results_individual}")
print(f"Group results: {path_results_group}")

# =============================================================================
# TFR PARAMETERS
# =============================================================================

freqs = np.arange(10, 41, 1)  # 10-40 Hz as requested
n_cycles = 2
time_bandwidth = 2.0

# Storage for group averages
all_tfr_multitaper = []
all_tfr_morlet = []

# =============================================================================
# PROCESS EACH PARTICIPANT
# =============================================================================

for subj in subjects:
    print(f"\n=== Processing {subj} ===")
    
    # Paths for current subject
    path_task_data_raw = os.path.join(path_data, subj, 'raw_rotated_sensorlocations', f"{subj}_file-oddballTones_upright_wsensorlocations_raw.fif")
    path_bad_channels = os.path.join(path_data, subj, f"{subj}_badchannels.tsv")
    path_event_dictionary = os.path.join(path_data, subj, f"{subj}_event_dict.json")
    
    # Check if files exist
    if not os.path.exists(path_task_data_raw):
        print(f"Skipping {subj}: raw data file not found")
        continue
    
    # =============================================================================
    # LOAD DATA
    # =============================================================================
    
    raw = mne.io.read_raw_fif(path_task_data_raw, preload=True)
    
    # =============================================================================
    # BAD CHANNELS
    # =============================================================================
    
    if os.path.exists(path_bad_channels):
        bad_channels = pd.read_csv(path_bad_channels, sep='\t')
        bad_channels = bad_channels['badchannelslots'].tolist()
        print(f"Bad channels for {subj}: {bad_channels}")
        
        raw.info['bads'].clear()
        raw.info['bads'].extend(bad_channels)
    else:
        print(f"No bad channels file for {subj}")
    
    # =============================================================================
    # EVENTS
    # =============================================================================
    
    events = mne.find_events(raw, stim_channel='di32')
    
    # =============================================================================
    # EVENT DICTIONARY
    # =============================================================================
    
    if os.path.exists(path_event_dictionary):
        with open(path_event_dictionary, 'r') as f:
            event_dict = json.load(f)
    else:
        # Default event dictionary
        event_dict = {'freq/tone': 1, 'infreq/tone': 2}
        print(f"Using default event dict for {subj}: {event_dict}")
    
    print(f"Event dictionary: {event_dict}")
    
    # =============================================================================
    # EPOCHS
    # =============================================================================
    
    epochs = mne.Epochs(raw,
                events, event_dict,
                tmin=-0.3, tmax=0.6,
                baseline=None,
                proj=False,
                picks='all',
                detrend=1,
                reject_by_annotation=True,
                preload=True,
                verbose=False)
    
    print(f"Created epochs: {epochs}")
    
    # =============================================================================
    # EQUALIZE EPOCH NUMBERS BTW CONDITIONS
    # =============================================================================
    
    event_id = event_dict
    epochs_list = [epochs[k] for k in event_id]
    
    # Equalize and combine epochs
    equalized_epochs = equalize_and_combine_epochs(epochs_list, seed=42)
    print(f"Equalized epochs: {equalized_epochs}")
    
    # =============================================================================
    # TIME-FREQUENCY ANALYSIS
    # =============================================================================
    
    # Multitaper TFR
    print(f"Computing multitaper TFR for {subj}...")
    tfr_multitaper = equalized_epochs.compute_tfr(
        method="multitaper",
        freqs=freqs, 
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, 
        picks='mag',
        use_fft=True, 
        return_itc=False,
        average=True, 
        decim=1,
        n_jobs=-1)
    
    # Morlet TFR
    print(f"Computing morlet TFR for {subj}...")
    tfr_morlet = equalized_epochs.compute_tfr(
        method="morlet",
        freqs=freqs, 
        n_cycles=n_cycles,  # Adaptive cycles for morlet
        picks='mag',
        use_fft=True, 
        return_itc=False,
        average=True, 
        decim=1,
        n_jobs=-1)
    
    # Store for group average
    all_tfr_multitaper.append(tfr_multitaper)
    all_tfr_morlet.append(tfr_morlet)
    
    # =============================================================================
    # PLOT INDIVIDUAL PARTICIPANT (both methods separately)
    # =============================================================================
    
    # Plot multitaper
    fig1 = tfr_multitaper.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'{subj} - Multitaper TFR - 10-40Hz, -0.1 to 0.5s, baseline -0.1 to 0s',
        show=False)
    
    # Save multitaper plot
    filename1 = f"{subj}_tfr_multitaper_10-40Hz_-0.1-0.5s.png"
    filepath1 = os.path.join(path_results_individual, filename1)
    fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot morlet
    fig2 = tfr_morlet.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'{subj} - Morlet TFR - 10-40Hz, -0.1 to 0.5s, baseline -0.1 to 0s',
        show=False)
    
    # Save morlet plot
    filename2 = f"{subj}_tfr_morlet_10-40Hz_-0.1-0.5s.png"
    filepath2 = os.path.join(path_results_individual, filename2)
    fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Saved: {filename1}")
    print(f"Saved: {filename2}")
    
    # Create combined figure manually for comparison
    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create separate plots and combine them
    fig_temp1 = tfr_multitaper.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'Multitaper',
        show=False)
    
    fig_temp2 = tfr_morlet.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'Morlet',
        show=False)
    
    plt.close(fig_temp1)
    plt.close(fig_temp2)
    plt.close(fig_combined)
    
    # Create a simple comparison text file instead
    comparison_info = f"""
Subject: {subj}
TFR Analysis Comparison
-----------------------
Frequency range: 10-40 Hz
Time window: -0.1 to 0.5 s
Baseline: -0.1 to 0 s
Methods: Multitaper and Morlet

Files generated:
- {filename1}
- {filename2}
"""
    
    info_file = os.path.join(path_results_individual, f"{subj}_tfr_analysis_info.txt")
    with open(info_file, 'w') as f:
        f.write(comparison_info)

# =============================================================================
# GROUP AVERAGES
# =============================================================================

print(f"\n=== Computing Group Averages (N={len(all_tfr_multitaper)}) ===")

if len(all_tfr_multitaper) == 0:
    print("No valid subjects found! Check your data paths.")
else:
    # Compute grand averages
    tfr_multitaper_ga = mne.grand_average(all_tfr_multitaper)
    tfr_morlet_ga = mne.grand_average(all_tfr_morlet)
    
    # Plot group averages separately
    
    # Group multitaper
    fig1 = tfr_multitaper_ga.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'Group Average (N={len(all_tfr_multitaper)}) - Multitaper TFR - 10-40Hz, -0.1 to 0.5s',
        show=False)
    
    # Save group multitaper
    group_filename1 = f"group_average_N{len(all_tfr_multitaper)}_multitaper_10-40Hz.png"
    group_filepath1 = os.path.join(path_results_group, group_filename1)
    fig1.savefig(group_filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Group morlet
    fig2 = tfr_morlet_ga.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'Group Average (N={len(all_tfr_morlet)}) - Morlet TFR - 10-40Hz, -0.1 to 0.5s',
        show=False)
    
    # Save group morlet
    group_filename2 = f"group_average_N{len(all_tfr_morlet)}_morlet_10-40Hz.png"
    group_filepath2 = os.path.join(path_results_group, group_filename2)
    fig2.savefig(group_filepath2, dpi=300, bbox_inches='tight')
    plt.show()  # Show the last plot
    
    print(f"Saved group averages:")
    print(f"  Multitaper: {group_filename1}")
    print(f"  Morlet: {group_filename2}")
    
    # Save TFR objects for future use
    tfr_multitaper_ga.save(os.path.join(path_results_group, f"group_multitaper_N{len(all_tfr_multitaper)}.h5"), overwrite=True)
    tfr_morlet_ga.save(os.path.join(path_results_group, f"group_morlet_N{len(all_tfr_morlet)}.h5"), overwrite=True)
    
    print("Analysis complete!")
    print(f"Individual plots: {path_results_individual}")
    print(f"Group averages: {path_results_group}")