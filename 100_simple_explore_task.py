#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:44:32 2025

@author: b.pomiechowska@bham.ac.uk; PLEASE ADD YOUR EMAILS HERE!
"""

import numpy as np
np.alltrue = np.all
import matplotlib 
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

import matplotlib.pyplot as plt

import mne
import os
import pandas as pd
import json

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm/'

# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

# Enter which participant you would like to explore
subj = 'sub-107'

# Enter which preprocessing routine would you like to explore
# Select the input from: processed_1_filter, processed_2_filter_ica, processed_3_filter_ica_manualclean
preprocessing_routine_input = 'processed_3_filter_ica_manualclean';

# Set output conditionally based on selected input
if preprocessing_routine_input == 'processed_1_filter':
    preprocessing_routine_output = 'preprocessing_routine_1'

elif preprocessing_routine_input == 'processed_2_filter_ica':
    preprocessing_routine_output = 'preprocessing_routine_2'

elif preprocessing_routine_input == 'processed_3_filter_ica_manualclean':
    preprocessing_routine_output = 'preprocessing_routine_3'

else:
    raise ValueError('Unknown preprocessing_routine_input')


# =============================================================================
# PATHS
# =============================================================================

# Construct paths
path_data  = os.path.join(root_data_path,'data')
path_results_erf  = os.path.join(root_data_path,'results',preprocessing_routine_output,'erf')
path_results_rms  = os.path.join(root_data_path,'results',preprocessing_routine_output,'rms')

print(path_data)
print(path_results_erf)
print(path_results_rms)

path_evoked = os.path.join(path_data,subj,'evoked',f"{subj}_evoked.fif")

print(path_evoked)

# Path task
path_task_data_raw = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-oddballTones_upright_wsensorlocations_raw.fif")
#path_task_data_filtered = os.path.join(path_data,subj,'processed_filtered',f"{subj}_file-oddballTones_filtered_01_40.fif")
#path_task_data_preprocessing_routine_3 = os.path.join(path_data,subj,'preprocessing_routine_2',f"{subj}_manual_clean.fif")
path_bad_channels = os.path.join(path_data,subj,f"{subj}_badchannels.tsv")
path_event_dictionary = os.path.join(path_data,subj,f"{subj}_event_dict.json")

print(path_task_data_raw)
print(path_bad_channels)
print(path_event_dictionary)

path_load_data = path_task_data_preprocessing_routine_3

# =============================================================================
# LOAD DATA (raw & filtered)
# =============================================================================

raw = mne.io.read_raw_fif(path_task_data_raw, preload=True)
fildata = mne.io.read_raw_fif(path_load_data, preload=True)

# =============================================================================
# CROP DATA FILE [if needed]
# =============================================================================

#short = raw.copy().crop(tmin = 0, tmax = 260)

# =============================================================================
# BAD CHANNELS (fildata)
# =============================================================================

# Load bad channel info from the csv file
bad_channels = pd.read_csv(path_bad_channels, sep='\t')
bad_channels = bad_channels['badchannelslots'].tolist()
print(bad_channels)

fildata.info['bads'].clear()
# Add bad channels to raw.info['bads']
fildata.info['bads'].extend(bad_channels)

print("Bad channels set in fildata:", raw.info['bads'])


# =============================================================================
# PSD
# =============================================================================

n_fft = 10000
rawpsd = raw.compute_psd(method="welch", fmin=.1, fmax=125, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
rawpsd.plot()

filpsd = fildata.compute_psd(method="welch", fmin=.1, fmax=125, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
filpsd.plot()

# =============================================================================
# FILTER DATA % uncomment if need for filtering
# =============================================================================

# fildata = raw.copy()
# fildata.filter(.1, 40)
# fildata.notch_filter(np.arange(50, 251, 50), notch_widths=5)

# =============================================================================
# EVENTS
# =============================================================================

events = mne.find_events(raw, stim_channel='di32')
# mne.viz.plot_events(events)

# =============================================================================
# EVENT DICTIONARY
# =============================================================================

with open(path_event_dictionary, 'r') as f:
    event_dict = json.load(f)

print(event_dict)

# =============================================================================
# COMPUTE & PLOT ISI
# =============================================================================

# Plot the event channel
#aux = raw.copy().pick(picks="stim").plot(start=3, duration=6)

# Calculate the ISI (i.e. time between events)
differences_samples = np.diff(events[:, 0])
differences = np.diff(events[:, 0])*0.0002
differences_hist = differences[differences < 1]

# Plot the ISI
plt.hist(differences_hist, bins=50, color='skyblue', edgecolor='black')

print(differences_samples[:10])
print(differences[:10])

# =============================================================================
# EPOCHS
# =============================================================================
epochs = mne.Epochs(fildata, events, event_id=event_dict, tmin=-0.1, tmax=.6, detrend=1, reject_by_annotation=True)

print(epochs)

# =============================================================================
# PLOT EPOCHS - BUTTERFLY
# =============================================================================

#epochs.plot(n_epochs=10, events = True)

ylim = dict(mag=[-500, 1000])
times = np.linspace(-.1, .6, 5)

evoked_overall = epochs.average()
fig_evoked_simple = evoked_overall.plot(spatial_colors=True, gfp=True, picks = "mag", exclude = 'bads')
fig_evoked_joint = evoked_overall.plot_joint(times=times)

fig_evoked_joint.suptitle("ERF (data collapsed across conditions): " + subj)
fig_evoked_joint.savefig(os.path.join(path_results_erf,subj+'_erf_joint_overall'), dpi=300, bbox_inches="tight")

fig_evoked_simple.suptitle("ERF (data collapsed across conditions): " + subj)
fig_evoked_simple.savefig(os.path.join(path_results_erf,subj+'_erf_simple_overall'), dpi=300, bbox_inches="tight")

fig_evoked_topo = evoked_overall.plot_topomap(ch_type="mag", times=times, colorbar=True, average=0.05) # vlim=(-500, 500)

fig_evoked_topo.suptitle("ERF (data collapsed across conditions): " + subj)
fig_evoked_topo.savefig(os.path.join(path_results_erf,subj+'_erf_topo_overall'), dpi=300, bbox_inches="tight")


# =============================================================================
# EQUALIZE EPOCH NUMBERS BTW CONDITIONS (FREQ vs INFREQ)
# =============================================================================
event_id = event_dict

epochs_list = [epochs[k] for k in event_id]
#mne.epochs.equalize_epoch_counts(epochs_list, method='random')

# =============================================================================
# EQUALIZE RANDOMLY EPOCH NUMBERS BTW CONDITIONS (FREQ vs INFREQ)
# =============================================================================

# Concatenate the epochs
combined_epochs = mne.concatenate_epochs(epochs_list)

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

# Usage
equalized_epochs = equalize_and_combine_epochs(epochs_list, seed=42)

print(equalized_epochs)

# =============================================================================
# PLOT EPOCHS w equal numbers per condition
# =============================================================================

aux = equalized_epochs

#ts_args = dict(ylim=dict(mag=[-850, 850]))

evoked_frequent = aux["freq/tone"].average()
freq = evoked_frequent.plot_joint(title = "freq/tone", times = times)

evoked_infrequent = aux["infreq/tone"].average()
infreq = evoked_infrequent.plot_joint(title = "infreq/tone",times = times)

freq.suptitle("ERF frequent tone: " + subj)
freq.savefig(os.path.join(path_results_erf,subj+'_erf_joint_freq'), dpi=300, bbox_inches="tight")

infreq.suptitle("ERF infrequent tone: " + subj)
infreq.savefig(os.path.join(path_results_erf,subj+'_erf_joint_infreq'), dpi=300, bbox_inches="tight")

# =============================================================================
# SAVE EVOKED 
# =============================================================================

evokeds = [evoked_overall, evoked_frequent, evoked_infrequent]
mne.write_evokeds(path_evoked, evokeds, overwrite = True)

# =============================================================================
# ROOT MEAN SQUARE
# =============================================================================

evokeds = {
    'freq/tone': aux['freq/tone'].average(),
    'infreq/tone': aux['infreq/tone'].average()
}

fig = mne.viz.plot_compare_evokeds(
    evokeds,
    picks='mag',
    ci=None,                     # ±SEM shading
    colors={'freq/tone':'C0',    # customize colors if you like
            'infreq/tone':'C1'},
    title='Freq vs. InFreq (mag channels)',
    show_sensors='upper right',  # show sensor layout inset
    show=False                   # don't pop up GUI if scripting
)

fig[0].suptitle("RMS: " + subj)
fig[0].savefig(os.path.join(path_results_rms,subj+'_rms'), dpi=300, bbox_inches="tight")

