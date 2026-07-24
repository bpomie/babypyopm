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

# Inser the path to your project folder
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

# Enter which participant you would like to explore
subj = 'sub-101'

# =============================================================================
# PATHS
# =============================================================================

# Construct paths

# Construct paths
path_data  = os.path.join(root_data_path,'data')
path_results_tfr  = os.path.join(root_data_path,'results','_file-oddballTones_processed_2_filter_ica','tfr')

print(path_data)
print(path_results_tfr)

path_epo_tfr = os.path.join(path_data,subj,'epo_tfr',f"{subj}_epo-tfr.h5")
path_avg_tfr = os.path.join(path_data,subj,'avg_tfr',f"{subj}_avg-tfr.h5")

# Path task
path_task_data_raw = os.path.join(path_data,subj,'processed_2_filter_ica',f"{subj}_file-oddballTones_processed_2_filter_ica.fif")
path_bad_channels = os.path.join(path_data,subj,f"{subj}_badchannels.tsv")
path_event_dictionary = os.path.join(path_data,subj,f"{subj}_event_dict.json")

path_load_data = path_task_data_raw

# =============================================================================
# LOAD DATA (raw & filtered)
# =============================================================================
raw = mne.io.read_raw_fif(path_load_data, preload=True)

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

raw.info['bads'].clear()
# Add bad channels to raw.info['bads']
raw.info['bads'].extend(bad_channels)

print("Bad channels set in raw data:", raw.info['bads'])

# =============================================================================
# EVENTS
# =============================================================================

events = mne.find_events(raw, stim_channel='di32')

# =============================================================================
# EVENT DICTIONARY
# =============================================================================

with open(path_event_dictionary, 'r') as f:
    event_dict = json.load(f)

print(event_dict)

# =============================================================================
# EPOCHS
# =============================================================================
epochs = mne.Epochs(raw,
            events, event_dict,
            tmin=-0.3, tmax=0.6,
            baseline=None,
            proj=False,
            picks = 'all',
            detrend = 1,
            reject_by_annotation=True,
            preload=True,
            verbose=False)

print(epochs)

# =============================================================================
# EQUALIZE EPOCH NUMBERS BTW CONDITIONS (FREQ vs INFREQ)
# =============================================================================
event_id = event_dict
epochs_list = [epochs[k] for k in event_id]

# Usage
equalized_epochs = equalize_and_combine_epochs(epochs_list, seed=42)

print(equalized_epochs)

# =============================================================================
# TIME-FREQUENCY ANALYSIS
# =============================================================================


freqs = np.arange(8, 40, 2)
n_cycles = 2
time_bandwidth = 2.0

tfr_all = equalized_epochs.compute_tfr(
    method = "multitaper",
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'mag',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=1,
    n_jobs = -1)

# Plot overall TFR
tfr_all.plot_topo(
    tmin=-.1, tmax=.5, 
    baseline=[-.1,0], 
    mode="percent", 
    vmin=-0.5, vmax=0.5,
    fig_facecolor='w',
    font_color='k',
    title='TFR of power <40 Hz')

# tfr_infreq =  equalized_epochs['infreq'].compute_tfr(
#     method = "multitaper",
#     freqs=freqs, 
#     n_cycles=n_cycles,
#     time_bandwidth=time_bandwidth, 
#     picks = 'mag',
#     use_fft=True, 
#     return_itc=False,
#     average=True, 
#     decim=1,
#     n_jobs = -1)

# tfr_freq =  equalized_epochs['freq'].compute_tfr(
#     method = "multitaper",
#     freqs=freqs, 
#     n_cycles=n_cycles,
#     time_bandwidth=time_bandwidth, 
#     picks = 'mag',
#     use_fft=True, 
#     return_itc=False,
#     average=True, 
#     decim=1,
#     n_jobs = -1)







#  Define frequency parameters
# freqs = np.logspace(*np.log10([1, 40]), num=40)  # Frequencies from 1 to 40 Hz
# n_cycles = freqs / 2.  # Number of cycles for each frequency

# # Compute TFR for all conditions combined
# print("Computing TFR for all conditions combined...")
# power_overall = mne.time_frequency.tfr_morlet(equalized_epochs, 
#                                               freqs=freqs, 
#                                               n_cycles=n_cycles, use_fft=True,
#                                              return_itc=False, decim=3, n_jobs=-1)

# # Plot overall TFR
# power_overall.plot_topo(
#     tmin=-.5, tmax=.6, 
#     baseline=[-0.5, -0.25], 
#     mode="percent", 
#     fig_facecolor='w',
#     font_color='k',
#     title='TFR of power <40 Hz')







# # fig_overall.savefig(os.path.join(path_results_tfr, f'{subj}_tfr_overall.png'), 
# #                    dpi=300, bbox_inches="tight")



# tfr_freq.plot_topomap(
#     tmin=0.5, tmax=1, 
#     fmin=9, fmax=11,
#     baseline=[-0.5,-0.25], 
#     mode="percent",
#     size=3
# )

