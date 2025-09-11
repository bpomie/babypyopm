#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:17:43 2025

@author: b.pomiechowska@bham.ac.uk
"""

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

import os

import utils_study 
study = utils_study.Study
utils = utils_study.Utils()

import utils_preprocessing_analysis
preprocess_analyse = utils_preprocessing_analysis.OPM_Pipeline(incl_report=False)

import mne

# =============================================================================
# Parameters 
# =============================================================================

input_folder = 'evoked'
output_folder = 'processed_filtered'

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm/'

path_results_rms  = os.path.join(root_data_path,'results','preprocessing_routine_3','erf_grandaverage')

# =============================================================================
# Set up 
# =============================================================================

# Get paths to all subfolders
paths = utils.get_paths(root_data_path)

# Define the template for generating the filename for raw FIF data files
basename = '{subj}/evoked/{sub-subj}_evoked.fif'

# Find FIF files that correspond to the template defined above
dataset = study(os.path.join(paths.data, basename))
 
# List all folders / subjects in '~/data'
subjects = [f for f in os.listdir(paths.data) if os.path.isdir(os.path.join(paths.data, f))]
print(subjects)

# Load evoked
"""Each evoked object contains three sets of epochs: 
    1 all epochs across frequent and infrequent conditions
    2 epochs from the frequent condition
    3 epochs from the infrequent
    """
all_subjects_evokeds = preprocess_analyse.load_evoked(dataset, subjects,paths.data,input_folder)


for subj_evokeds in all_subjects_evokeds:
    for evk in subj_evokeds:
        evk.pick_types(eeg=False, meg=True, misc=False)  # pick only relevant channels

# Transpose the list to group by condition
conditions_grouped = list(zip(*all_subjects_evokeds))

grand_averages = [mne.grand_average(cond_list) for cond_list in conditions_grouped]

titles = ['Overall', 'Frequent', 'Infrequent']

for i, evk in enumerate(grand_averages, start=0):
    fig = evk.plot_joint()
    fig.suptitle(titles[i], fontsize=14)
    plt.show()
    fig.savefig(os.path.join(path_results_rms,titles[i]+'_grand_average'), dpi=300, bbox_inches="tight")

