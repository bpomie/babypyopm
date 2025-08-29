
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:44:32 2025

@author: b.pomiechowska@bham.ac.uk

OUTPUT files
    .PNG with psd stored in ~/results/psd/

"""

import numpy as np
np.alltrue = np.all
import matplotlib 
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

import mne
import matplotlib.pyplot as plt
import os

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm'

# Construct paths
path_data  = os.path.join(root_data_path,'data')
path_results_psd  = os.path.join(root_data_path,'results','psd')

print(path_data)
print(path_results_psd)


# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

# Enter which participant you would like to explore
subj = 'sub-107'

# =============================================================================
# PATHS
# =============================================================================

# Compile the paths to both emptyroom and task recordings (i.e. raw data rotated w/ sensor locations)

# Path empty room
path_emptyroom_data = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-emptyroom_upright_wsensorlocations_raw.fif")

# Path task
path_task_data = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-oddballTones_upright_wsensorlocations_raw.fif")

print(path_emptyroom_data)
print(path_task_data)

# =============================================================================
# LOAD DATA
# =============================================================================

raw_emptyroom = mne.io.read_raw_fif(path_emptyroom_data, preload=True)
raw_task = mne.io.read_raw_fif(path_task_data, preload=True)

# =============================================================================
# PSD
# =============================================================================

n_fft = 10000

rawpsd_emptyroom = raw_emptyroom.compute_psd(method="welch", fmin=.1, fmax=125, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
rawpsd_task = raw_task.compute_psd(method="welch", fmin=.1, fmax=125, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))

fig_emptyroom = rawpsd_emptyroom.plot(picks = 'all')
fig_emptyroom.suptitle("Empty Room PSD: " + subj)
fig_task = rawpsd_task.plot()
fig_task.suptitle("Task PSD: " + subj)

fig_emptyroom.savefig(os.path.join(path_results_psd,subj+'_emptyroom'), dpi=300)
fig_task.savefig(os.path.join(path_results_psd,subj+'_task'), dpi=300)


# =============================================================================
# LIFTS
# =============================================================================

# aux = raw_task

# picks = mne.pick_types(aux.info, meg=True, exclude='bads')
 
# amp_scale = 1e12  # Converting to pico Tesla(pT)
# stop = len(aux.times)
# step = 1
# data_ds1, time_ds = aux[picks[::2], :stop] # picks[::5] -- selects every nth sensor;
# data_ds2, time_ds = data_ds1[:, ::step] * amp_scale, time_ds[::step]
# fig_raw_amp, ax = plt.subplots(layout="constrained")
# plot_kwargs = dict(lw=1, alpha=1)
# ax.plot(time_ds, data_ds2.T - np.mean(data_ds2, axis=1), **plot_kwargs)
# ax.grid(True)
# set_kwargs = dict(ylim=(-3000, 3000), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)")
# ax.set(title="Before Filters", **set_kwargs)
# plt.show()
