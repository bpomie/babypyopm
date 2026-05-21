#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:44:32 2025

@author: b.pomiechowska@bham.ac.uk
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

import utils_study 
study = utils_study.Study
utils = utils_study.Utils()


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

# =============================================================================
# SELECT PREPROCESSING ROUTINE
# =============================================================================

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
# Set up 
# =============================================================================

# Get paths to all subfolders
paths = utils.get_paths(root_data_path)

# Define the template for generating the filename for raw FIF data files
basename = f'{{subj}}/{preprocessing_routine_input}/{{sub-subj}}_evoked.fif'

# Find FIF files that correspond to the template defined above
dataset = study(os.path.join(paths.data, basename))
 
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


# Path evoked data
path_load_data = dataset.match_files[0]

print(path_load_data)

# =============================================================================
# LOAD DATA (evoked)
# =============================================================================

# Load evoked
"""Each evoked object contains three sets of epochs: 
    1 all epochs across frequent and infrequent conditions
    2 epochs from the frequent condition
    3 epochs from the infrequent
    Note that the number of epochs across frequent vs infrequent conditions is matched
    """

evoked_data = mne.read_evokeds(path_load_data)

# =============================================================================
# PLOT EPOCHS w equal numbers per condition
# =============================================================================

times = np.linspace(-.1, .6, 5)

evoked_overall = evoked_data[0]
overall = evoked_overall.plot_joint(title = "freq/tone", times = times)

evoked_frequent = evoked_data[1] 
freq = evoked_frequent.plot_joint(title = "freq/tone", times = times)

evoked_infrequent = evoked_data[2] 
infreq = evoked_infrequent.plot_joint(title = "infreq/tone",times = times)

freq.suptitle("ERF overall: " + subj)
freq.savefig(os.path.join(path_results_erf,subj+'_erf_joint_overall'), dpi=300, bbox_inches="tight")

freq.suptitle("ERF frequent tone: " + subj)
freq.savefig(os.path.join(path_results_erf,subj+'_erf_joint_freq'), dpi=300, bbox_inches="tight")

infreq.suptitle("ERF infrequent tone: " + subj)
infreq.savefig(os.path.join(path_results_erf,subj+'_erf_joint_infreq'), dpi=300, bbox_inches="tight")


# =============================================================================
# ROOT MEAN SQUARE
# =============================================================================


evokeds = {
    'freq/tone': evoked_data[1],
    'infreq/tone': evoked_data[2]
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

