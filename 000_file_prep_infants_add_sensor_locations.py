"""
The raw data from the FieldLine rigid infant helmet do not contain sensor locations.

This script prepares the datafiles for analysis in MNE.

It reads in sensor locations and rotates the montage for plotting in MNE (from supine postition used during recording to upright/sitting position).

1. Load the raw data for the task of choice (e.g. oddballTones, emptyroom)
2. Add sensor locations & annotate which sensors were reference sensors
3. Rotate sensors from supine to upright/sitting position for plotting
4. Plot and save the montage 
5. Save the .fif file w/ rotated montage and sensors locations

INPUT FILES:
    meg files (.FIF) located in ~/data/{sub-subj}/raw_rotated_sensorlocations/

OUTPOUT FILES 
    montage plots (.PNG) to ~/montages/
    meg files (.FIF) {sub-subj}_file-{task}_upright_wsensorlocations_raw.fif to ~/data/{sub-subj}/raw_rotated_sensorlocations/
"""

import os

import utils_study 
study = utils_study.Study
utils = utils_study.Utils()

import utils_infant_helmet 
helmet = utils_infant_helmet.SensorLayout()

import utils_preprocessing_analysis
preprocess_analyse = utils_preprocessing_analysis.OPM_Pipeline(incl_report=False)

# =============================================================================
# Parameters 
# =============================================================================

task = 'oddballTones' #'oddballTones'
output_filename = task + '_'+'upright_wsensorlocations_raw'
ouput_folder = 'raw_rotated_sensorlocations'
output_montage_folder = 'montages'

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/analysesopm/project_setup_methods_paper'

# =============================================================================
# Set up 
# =============================================================================

# Get paths to all subfolders
paths = utils.get_paths(root_data_path)

# Define the template for generating the filename for raw FIF data files
basename = '{subj}/raw_recording/{date}_{time}_{sub-subj}_file-{task}_raw.fif'

# Find FIF files that correspond to the template defined above
dataset = study(os.path.join(paths.data, basename))
 
# List all folders / subjects in '~/data'
subjects = [f for f in os.listdir(paths.data) if os.path.isdir(os.path.join(paths.data, f))]
print(subjects)

# =============================================================================
# LOAD DATA
# =============================================================================

# If you provide n>1 task names (e.g., tasks = ['oddballTones','oddballSyllables]), then these data will be concatenated into one output files 
tasks = [task]
raw = utils.load_concatenate_data(dataset, subjects, tasks)

# =============================================================================
# ADD SENSOR LOCATIONS, ROTATE THE MONTAGE (from supine to upright position), PLOT & SAVE THE MONTAGE PLOT
# =============================================================================

# Add all sensor locations, annotate which sensors were reference sensors
data_sensor_loc = [helmet.add_sensor_locations(d, dataset, task, paths.data) for d in raw]

# Rotate sensors from supine to upright/sitting position
data_aligned = [helmet.align_topo_locs(d) for d in data_sensor_loc]

# Plot montage + save montage plots for each subj to /results/
path_save_montage = os.path.join(root_data_path,output_montage_folder)
[preprocess_analyse.plot_montage(d,path_save_montage) for d in data_aligned] 

[preprocess_analyse.plot_montage_3D(d,path_save_montage) for d in data_aligned] 

# =============================================================================
# SAVE DATA
# =============================================================================

# Save fif files with sensor locations

[utils.save_fif(d, output_filename, ouput_folder, paths.data, timestmp=False, ) for d in data_aligned]

