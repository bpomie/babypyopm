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

input_folder = 'raw_rotated_sensorlocations'
output_folder = 'processed_filtered'
task = 'oddballTones'

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm/'

# =============================================================================
# Set up 
# =============================================================================

# Get paths to all subfolders
paths = utils.get_paths(root_data_path)

# Define the template for generating the filename for raw FIF data files
basename = '{subj}/raw_rotated_sensorlocations/{sub-subj}_file-{task}_raw.fif'

# Find FIF files that correspond to the template defined above
dataset = study(os.path.join(paths.data, basename))
 
# List all folders / subjects in '~/data'
subjects = [f for f in os.listdir(paths.data) if os.path.isdir(os.path.join(paths.data, f))]
print(subjects)
subjects = ['sub-101']

# Load datasets
data = preprocess_analyse.load_data(dataset, subjects,paths.data,input_folder,task)

mne.viz.plot_sensors(data[0].info, show_names=True)

filtered_data = [preprocess_analyse.run_filter(d, .1, 40, notch = True) for d in data]

mne.viz.plot_sensors(filtered_data[0].info, show_names=True)

[utils.save_fif_path(d, task,'filtered_01_40', paths.data, output_folder) for d in filtered_data]




