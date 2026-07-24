#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Frequency Analysis Script with Individual Sensor Plots
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


# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Insert the path to your project folder
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SENSOR LIST FOR INDIVIDUAL ANALYSIS
# =============================================================================

# Define the list of sensors you want to analyze individually
# Modify this list according to your specific sensor names
sensors_of_interest = [
    'L38', 'L46', 'L40', 'L31', 'L30',
    'R38', 'R46', 'R40', 'R31', 'R30'
]

# Alternative: You can also specify sensors by their original channel names
# sensors_of_interest = ['s12_bz', 's24_bz', 's36_bz']

print(f"Sensors of interest: {sensors_of_interest}")

# =============================================================================
# PARTICIPANT LIST
# =============================================================================

# List all subjects in the data folder
path_data = os.path.join(root_data_path, 'data')
preproc_folder='processed_2_filter_ica'
preproc_naming='_file-oddballTones_processed_2_filter_ica.fif'
subjects = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f)) and f.startswith('sub-')]
print(f"Found subjects: {subjects}")

# Alternatively, manually specify subjects:
# subjects = ['sub-101', 'sub-102', 'sub-107']

# =============================================================================
# PATHS
# =============================================================================

# Results paths
path_results_tfr = os.path.join(root_data_path, 'results', 'tfr_analysis', preproc_folder)
path_results_topo_individual = os.path.join(path_results_tfr, 'topo_individual')
path_results_topo_group = os.path.join(path_results_tfr, 'topo_group_average')
path_results_sensors_individual = os.path.join(path_results_tfr, 'sensors_individual')

# Create directories
os.makedirs(path_results_topo_individual, exist_ok=True)
os.makedirs(path_results_topo_group, exist_ok=True)
os.makedirs(path_results_sensors_individual, exist_ok=True)

print(f"Topographic individual results: {path_results_topo_individual}")
print(f"Topographic group results: {path_results_topo_group}")
print(f"Individual sensors results: {path_results_sensors_individual}")

# =============================================================================
# TFR PARAMETERS
# =============================================================================

freqs = np.arange(10, 41, 1)  # 10-40 Hz as requested
n_cycles = 2
time_bandwidth = 2.0

# Storage for group averages
all_tfr_multitaper = []
all_tfr_morlet = []

# Storage for individual sensor group averages
sensor_tfr_multitaper = {sensor: [] for sensor in sensors_of_interest}
sensor_tfr_morlet = {sensor: [] for sensor in sensors_of_interest}

# =============================================================================
# PROCESS EACH PARTICIPANT
# =============================================================================

for subj in subjects:
    print(f"\n=== Processing {subj} ===")
    
    # Paths for current subject
    fif_filename=subj + preproc_naming
    path_task_data_raw = os.path.join(path_data, subj, preproc_folder, fif_filename)
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
    
    # Print epoch counts before equalization
    print(f"Epoch counts before equalization:")
    for i, (condition, epoch_subset) in enumerate(zip(event_id.keys(), epochs_list)):
        print(f"  {condition}: {len(epoch_subset)} epochs")
    
    # Equalize epoch counts using MNE's built-in function
    mne.epochs.equalize_epoch_counts(epochs_list, method='random')
    
    # Print epoch counts after equalization
    print(f"Epoch counts after equalization:")
    for i, (condition, epoch_subset) in enumerate(zip(event_id.keys(), epochs_list)):
        print(f"  {condition}: {len(epoch_subset)} epochs")
    
    # Combine equalized epochs
    equalized_epochs = mne.concatenate_epochs(epochs_list)
    print(f"Total equalized epochs: {len(equalized_epochs)}")
    
    # =============================================================================
    # TIME-FREQUENCY ANALYSIS - ALL SENSORS
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
        n_cycles=n_cycles,
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
    # SAVE TFR DATA IN PARTICIPANT FOLDER
    # =============================================================================
    
    # Create TFR directory in participant's data folder
    subject_tfr_data_dir = os.path.join(path_data, subj, 'tfr')
    os.makedirs(subject_tfr_data_dir, exist_ok=True)
    
    # Save TFR objects
    tfr_multitaper_filename = os.path.join(subject_tfr_data_dir, f"{subj}_tfr_multitaper_10-40Hz.h5")
    tfr_morlet_filename = os.path.join(subject_tfr_data_dir, f"{subj}_tfr_morlet_10-40Hz.h5")
    
    tfr_multitaper.save(tfr_multitaper_filename, overwrite=True)
    tfr_morlet.save(tfr_morlet_filename, overwrite=True)
    
    print(f"Saved TFR data:")
    print(f"  Multitaper: {tfr_multitaper_filename}")
    print(f"  Morlet: {tfr_morlet_filename}")
    
    # =============================================================================
    # INDIVIDUAL SENSOR ANALYSIS
    # =============================================================================
    
    print(f"Processing individual sensors for {subj}...")
    
    # Create subject-specific sensor directory
    subject_sensor_dir = os.path.join(path_results_sensors_individual, subj)
    os.makedirs(subject_sensor_dir, exist_ok=True)
    
    # Check which sensors are available in the data
    available_sensors = [sensor for sensor in sensors_of_interest if sensor in tfr_multitaper.ch_names]
    missing_sensors = [sensor for sensor in sensors_of_interest if sensor not in tfr_multitaper.ch_names]
    
    if missing_sensors:
        print(f"Warning: Missing sensors for {subj}: {missing_sensors}")
    
    print(f"Available sensors for {subj}: {available_sensors}")
    
    # Process each available sensor
    for sensor in available_sensors:
        try:
            # Extract single sensor TFR data
            sensor_tfr_multi = tfr_multitaper.copy().pick_channels([sensor])
            sensor_tfr_mor = tfr_morlet.copy().pick_channels([sensor])
            
            # Store for group averaging
            sensor_tfr_multitaper[sensor].append(sensor_tfr_multi)
            sensor_tfr_morlet[sensor].append(sensor_tfr_mor)
            
            # Plot individual sensor TFR - Multitaper
            fig_multi = plt.figure(figsize=(10, 6))
            sensor_tfr_multi.plot(
                picks=sensor,
                tmin=-0.1, tmax=0.5,
                baseline=(-0.1, 0),
                mode='percent',
                # vmin=-50, vmax=50,
                title=f'{subj} - {sensor} - Multitaper TFR',
                show=False,
                colorbar=True,
                axes=fig_multi.gca())
            
            # Save multitaper plot
            filename_multi = f"{subj}_{sensor}_multitaper_tfr.png"
            filepath_multi = os.path.join(subject_sensor_dir, filename_multi)
            fig_multi.savefig(filepath_multi, dpi=300, bbox_inches='tight')
            plt.close(fig_multi)
            
            # Plot individual sensor TFR - Morlet
            fig_mor = plt.figure(figsize=(10, 6))
            sensor_tfr_mor.plot(
                picks=sensor,
                tmin=-0.1, tmax=0.5,
                baseline=(-0.1, 0),
                mode='percent',
                # vmin=-50, vmax=50,
                title=f'{subj} - {sensor} - Morlet TFR',
                show=False,
                colorbar=True,
                axes=fig_mor.gca())
            
            # Save morlet plot
            filename_mor = f"{subj}_{sensor}_morlet_tfr.png"
            filepath_mor = os.path.join(subject_sensor_dir, filename_mor)
            fig_mor.savefig(filepath_mor, dpi=300, bbox_inches='tight')
            plt.close(fig_mor)
            
            print(f"  Saved {sensor}: {filename_multi}, {filename_mor}")
            
        except Exception as e:
            print(f"  Error processing sensor {sensor} for {subj}: {e}")
            continue
    
    # =============================================================================
    # PLOT INDIVIDUAL PARTICIPANT (all sensors topo)
    # =============================================================================
    
    # Plot multitaper
    fig1 = tfr_multitaper.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        # vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'{subj} - Multitaper TFR - 10-40Hz, -0.1 to 0.5s, baseline -0.1 to 0s',
        show=True)
    
    # Save multitaper plot
    filename1 = f"{subj}_tfr_multitaper_10-40Hz_-0.1-0.5s.png"
    filepath1 = os.path.join(path_results_topo_individual, filename1)
    fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot morlet
    fig2 = tfr_morlet.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        # vmin=-0.5, vmax=0.5,
        fig_facecolor='w',
        font_color='k',
        title=f'{subj} - Morlet TFR - 10-40Hz, -0.1 to 0.5s, baseline -0.1 to 0s',
        show=True)
    
    # Save morlet plot
    filename2 = f"{subj}_tfr_morlet_10-40Hz_-0.1-0.5s.png"
    filepath2 = os.path.join(path_results_topo_individual, filename2)
    fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Saved: {filename1}")
    print(f"Saved: {filename2}")

# =============================================================================
# GROUP AVERAGES - ALL SENSORS
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
        vmin=-0.3, vmax=0.3,
        fig_facecolor='w',
        font_color='k',
        title=f'Group Average (N={len(all_tfr_multitaper)}) - Multitaper TFR - 10-40Hz, -0.1 to 0.5s',
        show=True)
    
    # Save group multitaper
    group_filename1 = f"group_average_N{len(all_tfr_multitaper)}_multitaper_10-40Hz.png"
    group_filepath1 = os.path.join(path_results_topo_group, group_filename1)
    fig1.savefig(group_filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Group morlet
    fig2 = tfr_morlet_ga.plot_topo(
        tmin=-0.1, tmax=0.5, 
        baseline=[-0.1, 0], 
        mode="percent", 
        vmin=-0.3, vmax=0.3,
        fig_facecolor='w',
        font_color='k',
        title=f'Group Average (N={len(all_tfr_morlet)}) - Morlet TFR - 10-40Hz, -0.1 to 0.5s',
        show=True)
    
    # Save group morlet
    group_filename2 = f"group_average_N{len(all_tfr_morlet)}_morlet_10-40Hz.png"
    group_filepath2 = os.path.join(path_results_topo_group, group_filename2)
    fig2.savefig(group_filepath2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Saved group averages:")
    print(f"  Multitaper: {group_filename1}")
    print(f"  Morlet: {group_filename2}")

# =============================================================================
# GROUP AVERAGES - INDIVIDUAL SENSORS
# =============================================================================

print(f"\n=== Computing Individual Sensor Group Averages ===")

# Create group sensor directory
group_sensor_dir = os.path.join(path_results_sensors_individual, 'sensors_group')
os.makedirs(group_sensor_dir, exist_ok=True)

for sensor in sensors_of_interest:
    if len(sensor_tfr_multitaper[sensor]) > 0:
        print(f"Computing group average for sensor {sensor} (N={len(sensor_tfr_multitaper[sensor])})")
        
        try:
            # Compute group averages for this sensor
            sensor_ga_multi = mne.grand_average(sensor_tfr_multitaper[sensor])
            sensor_ga_mor = mne.grand_average(sensor_tfr_morlet[sensor])
            
            # Plot sensor group average - Multitaper
            fig_multi = plt.figure(figsize=(10, 6))
            sensor_ga_multi.plot(
                picks=sensor,
                tmin=-0.1, tmax=0.5,
                baseline=(-0.1, 0),
                mode='percent',
                # vmin=-50, vmax=50,
                title=f'Group Average (N={len(sensor_tfr_multitaper[sensor])}) - {sensor} - Multitaper TFR',
                show=True,
                colorbar=True,
                axes=fig_multi.gca())
            
            # Save sensor group multitaper plot
            filename_multi = f"group_average_{sensor}_multitaper_N{len(sensor_tfr_multitaper[sensor])}.png"
            filepath_multi = os.path.join(group_sensor_dir, filename_multi)
            fig_multi.savefig(filepath_multi, dpi=300, bbox_inches='tight')
            plt.close(fig_multi)
            
            # Plot sensor group average - Morlet
            fig_mor = plt.figure(figsize=(10, 6))
            sensor_ga_mor.plot(
                picks=sensor,
                tmin=-0.1, tmax=0.5,
                baseline=(-0.1, 0),
                mode='percent',
                # vmin=-50, vmax=50,
                title=f'Group Average (N={len(sensor_tfr_morlet[sensor])}) - {sensor} - Morlet TFR',
                show=True,
                colorbar=True,
                axes=fig_mor.gca())
            
            # Save sensor group morlet plot
            filename_mor = f"group_average_{sensor}_morlet_N{len(sensor_tfr_morlet[sensor])}.png"
            filepath_mor = os.path.join(group_sensor_dir, filename_mor)
            fig_mor.savefig(filepath_mor, dpi=300, bbox_inches='tight')
            plt.close(fig_mor)
            
            print(f"  Saved {sensor} group averages: {filename_multi}, {filename_mor}")
            
            # Save individual sensor group TFR data in results folder
            sensor_ga_multi.save(os.path.join(group_sensor_dir, f"{sensor}_group_multitaper_N{len(sensor_tfr_multitaper[sensor])}.h5"), overwrite=True)
            sensor_ga_mor.save(os.path.join(group_sensor_dir, f"{sensor}_group_morlet_N{len(sensor_tfr_morlet[sensor])}.h5"), overwrite=True)
            print(f"  Saved {sensor} group TFR data in results folder")
            
        except Exception as e:
            print(f"  Error computing group average for sensor {sensor}: {e}")
            continue
    else:
        print(f"No data available for sensor {sensor}")

# =============================================================================
# SAVE OVERALL GROUP TFR OBJECTS
# =============================================================================

if len(all_tfr_multitaper) > 0:
    # Save overall group TFR objects in results folder
    tfr_multitaper_ga.save(os.path.join(path_results_topo_group, f"group_multitaper_N{len(all_tfr_multitaper)}.h5"), overwrite=True)
    tfr_morlet_ga.save(os.path.join(path_results_topo_group, f"group_morlet_N{len(all_tfr_morlet)}.h5"), overwrite=True)
    print(f"Saved overall group TFR data in results folder")

# =============================================================================
# SUMMARY
# =============================================================================

print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Processed {len(all_tfr_multitaper)} subjects successfully")
print(f"\nResults saved to:")
print(f"  Individual topographic plots: {path_results_topo_individual}")
print(f"  Group topographic averages: {path_results_topo_group}")
print(f"  Individual sensor plots: {path_results_sensors_individual}")
print(f"  Group sensor averages: {group_sensor_dir}")
print(f"\nTFR Data (.h5 files) saved to:")
print(f"  Individual subject data: data/{{subject}}/tfr/")
print(f"  Group averages: {path_results_topo_group}")
print(f"\nSensors analyzed: {sensors_of_interest}")
print(f"Available sensors across subjects: {list(set().union(*[s.ch_names for s in all_tfr_multitaper]))}")
print("="*60)