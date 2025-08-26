#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 21:25:52 2025

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
import glob

# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

# Enter which participant you would like to explore
subj = 'sub-002'
# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Inser the path to your project folder
#root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm/'
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled'

# =============================================================================
# PATHS
# =============================================================================

path_data  = os.path.join(root_data_path,'data')
path_montages  = os.path.join(root_data_path,'montages')

# Path task
match_task = os.path.join(path_data,subj,'raw_recording',f"*_{subj}_file-oddballTones_raw.fif")
files_task = glob.glob(match_task)
print(files_task)
path_task_data_raw = files_task[0]
save_task = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-oddballTones_upright_wsensorlocations_raw.fif")

# Path emptyroom
match_emptyroom = os.path.join(path_data,subj,'raw_recording',f"*_{subj}_file-emptyroom_raw.fif")
files_emptyroom = glob.glob(match_emptyroom)
path_emptyroom_data_raw = files_emptyroom[0]
save_emptyroom = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-emptyroom_upright_wsensorlocations_raw.fif")


# Path sensor locations
sensor_locations_path = os.path.join(path_data,subj,f"{subj}_sensor_locations.csv") 

# Path reference sensors
reference_sensors_locations_path = os.path.join(path_data,subj,f"{subj}_referencechannels_location.json")

# PRINT PATHS
print("Task data raw FIF:", path_task_data_raw)
print("Empty room data FIF:", path_emptyroom_data_raw)
print("Sensor locations CSV:", sensor_locations_path)
print("Reference sensors JSON:", reference_sensors_locations_path)

# =============================================================================
# LOAD DATA
# =============================================================================

raw_task = mne.io.read_raw_fif(path_task_data_raw, preload=True)
raw_emptyroom = mne.io.read_raw_fif(path_emptyroom_data_raw, preload=True)

# =============================================================================
# EXPLORE CHANNELS
# =============================================================================

# Extract channel names 
channel_list_task = raw_task.ch_names
channel_list_empty = raw_emptyroom.ch_names

print(channel_list_task)
print(channel_list_empty)

# PRINT task and emptyroom channels side by side
print("Task Channels      | Emptyroom Channels")
print("-------------------|--------------------")
for task_ch, empty_ch in zip(channel_list_task, channel_list_empty):
    print(f"{task_ch:<10} | {empty_ch}")
 
# ! NOTE: 
# Infant helmet channel format: s12_bz 
# Smart helmet channel format: R114_bz-s66
# Stimulus channel: di32  
 
# =============================================================================
# ASSIGN REFERENCE CHANNELS
# =============================================================================

print(raw_task.info)
print(raw_emptyroom.info)

# PART 1: Assign reference channels based on the smart helmet 
# i.e. assign smart channels, if any recorded, to be reference sensors

# Store smart channels detected in the empty room recording
list_smart_channels = [ch for ch in channel_list_empty if ch.startswith(('R', 'L'))]

# Set channel type to 'ref_meg' 
# for all channels located in the smart helmet across task and empty room recordings

if len(list_smart_channels) > 0:
    
    # Find matching sensors in the task recording and store their names
    matches = []
    for smart_ch in list_smart_channels:
        sensor_id = smart_ch.split('-')[-1] + "_bz"  # e.g. "s69_bz"
        if sensor_id in channel_list_task:
            matches.append((smart_ch, sensor_id))

    matching_sensor_ids = [m[1] for m in matches]
    print(matching_sensor_ids)
   
    # Empty room: build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
    type_map = {ch: 'ref_meg' for ch in list_smart_channels}
    raw_emptyroom.set_channel_types(type_map) # Set channel type
    # Task: build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
    type_map_task = {ch: 'ref_meg' for ch in matching_sensor_ids}
    raw_task.set_channel_types(type_map_task) # Set channel type

print(raw_task.info) 
print(raw_emptyroom.info)

# PART 2: Assign reference channels based on the JSON recording

if os.path.isfile(reference_sensors_locations_path):
    print("There is a _referencechannels_location JSON file.")
    # Read in the '_referencechannels_location.json' file
    with open(reference_sensors_locations_path, 'r') as f:
        reference_channels_info = json.load(f)
    # Store channel names 
    channel_list_json = [name for name, info in reference_channels_info.items() if info['position'] == [0, 0, 0]]
    print(channel_list_json)
    # Build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
    type_map = {ch: 'ref_meg' for ch in channel_list_json}
    raw_emptyroom.set_channel_types(type_map)
    raw_task.set_channel_types(type_map)

print(raw_task.info) 
print(raw_emptyroom.info)

# =============================================================================
# ADD SENSOR LOCATIONS  
# =============================================================================
    
sensor_locations = pd.read_csv(sensor_locations_path)
sensor_locations = sensor_locations.dropna(subset=['channel_name'])

# Create the new column 'ch_name' that can be used for averaging over the same slots accross participants
sensor_locations['slot_name'] = (
    sensor_locations['side'].str[0] +  # First character of 'side'
    sensor_locations['slot'].astype(str).str.zfill(2)   # Slot with leading zero
)
print(sensor_locations)

channel_names_csv = sensor_locations['channel_name'].tolist()



print(channel_names_csv)
len(channel_list_task)
len(channel_names_csv)

df = sensor_locations

"""
Notes:
    channel_info = raw.info['chs'][0]
    channel_info['ch_name']: channel name (e.g. 's24_bz')
    channel_info['kind']: channel type (e.g. FIFFV_MEG_CH = 1 for MEG)
    channel_info['loc'][:3]: 3D position (X, Y, Z)
    
    ch['loc'][3:6]: X-axis orientation vector
    ch['loc'][6:9]: Y-axis orientation vector
    ch['loc'][9:12]: Z-axis orientation vector
"""

for ch in raw_task.info['chs']:
    ch_name = ch['ch_name']
    print(ch)
    print(ch_name)

    # Store and print old position
    old_pos = ch['loc'][:3].copy() # recording_loc

    if ch_name in df['channel_name'].values:
        # Update the position based on CSV data
        pos = df[df['channel_name'] == ch_name][['X', 'Y', 'Z', 'x_i',
                                                 'x_j', 'x_k', 'y_i', 'y_j', 'y_k', 'z_i', 'z_j', 'z_k']].values[0]
        
        ch['loc'][:] = pos
        
for ch in raw_emptyroom.info['chs']:
    ch_name = ch['ch_name']
    print(ch)
    print(ch_name)

    if ch_name in df['channel_name'].values:
        # Update the position based on CSV data
        pos = df[df['channel_name'] == ch_name][['X', 'Y', 'Z', 'x_i',
                                                 'x_j', 'x_k', 'y_i', 'y_j', 'y_k', 'z_i', 'z_j', 'z_k']].values[0]
        
        ch['loc'][:] = pos

# RENAME SENSORS TO INCLUDE SLOT as ch_name
raw_task.annotations.custom_info = {'layout_info': sensor_locations}


for ch in raw_task.info['chs']:
    ch_name = ch['ch_name']
    index = raw_task.info['ch_names'].index(ch_name)
    print(ch_name)
    print(index)
    
    if ch_name in df['channel_name'].values:
        new_ch_name= df[df['channel_name'] == ch_name]['slot_name'].values[0]
        print(ch_name)
        print(new_ch_name)
        ch['ch_name']=new_ch_name
        # ch['sns_name']=ch_name
        raw_task.info['ch_names'][index]=new_ch_name
        

    
# PREVIEW SENSOR PLOTS to see what we're at
mne.viz.plot_sensors(raw_task.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_task.info, show_names=True, kind='topomap')

mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='topomap')

# =============================================================================
# CONVERT FROM MMs to Ms
# =============================================================================
    
for ch in raw_task.info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH:
        ch['loc'][:3] = ch['loc'][:3] / 1000  # convert mm → m
        
for ch in raw_emptyroom.info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH:
        ch['loc'][:3] = ch['loc'][:3] / 1000  # convert mm → m        

# PREVIEW SENSOR PLOTS to see what we're at

mne.viz.plot_sensors(raw_task.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_task.info, show_names=True, kind='topomap')

mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='topomap')

# =============================================================================
# ROTATE CHANNELS 180 DEGREES AROUND Z AXIS
# =============================================================================

# Equivalent 180° Z-axis rotation matrix
rotation_matrix = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1]
])

# Apply rotation to all MEG sensor positions
for ch in raw_task.info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH:
        loc_p = ch['loc'][:3]  # Extract (x, y, z) position
        loc_x_ijk = ch['loc'][3:6]  # Extract (x_i, x_j, x_k) 
        loc_y_ijk = ch['loc'][6:9]  # Extract (y_i, y_j, y_k) 
        loc_z_ijk = ch['loc'][9:12]  # Extract (z_i, z_j, z_k) 
        print(ch['loc'])
  
        #TO DO: Improve code elegance here. Can be done in a loop with location labels (e.g. x_j) assigned to dictionairy to improve robustness 
        rotated_loc_p = np.dot(loc_p, rotation_matrix)
        ch['loc'][:3] = rotated_loc_p
        
        rotated_loc_x_ijk = np.dot(loc_x_ijk, rotation_matrix)
        ch['loc'][3:6] = rotated_loc_x_ijk
        
        rotated_loc_y_ijk = np.dot(loc_y_ijk, rotation_matrix)
        ch['loc'][6:9] = rotated_loc_y_ijk
        
        rotated_loc_z_ijk = np.dot(loc_z_ijk, rotation_matrix)
        ch['loc'][9:12] = rotated_loc_z_ijk
             
        
# Apply rotation to all MEG sensor positions
for ch in raw_emptyroom.info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH:
        loc_p = ch['loc'][:3]  # Extract (x, y, z) position
        loc_x_ijk = ch['loc'][3:6]  # Extract (x_i, x_j, x_k) 
        loc_y_ijk = ch['loc'][6:9]  # Extract (y_i, y_j, y_k) 
        loc_z_ijk = ch['loc'][9:12]  # Extract (z_i, z_j, z_k) 
        print(ch['loc'])

         #TO DO: Improve code elegance here. Can be done in a loop with location labels (e.g. x_j) assigned to dictionairy to improve robustness
        rotated_loc_p = np.dot(loc_p, rotation_matrix)
        ch['loc'][:3] = rotated_loc_p
        
        rotated_loc_x_ijk = np.dot(loc_x_ijk, rotation_matrix)
        ch['loc'][3:6] = rotated_loc_x_ijk
        
        rotated_loc_y_ijk = np.dot(loc_y_ijk, rotation_matrix)
        ch['loc'][6:9] = rotated_loc_y_ijk
        
        rotated_loc_z_ijk = np.dot(loc_z_ijk, rotation_matrix)
        ch['loc'][9:12] = rotated_loc_z_ijk
             
# PREVIEW SENSOR PLOTS to see what we're at

mne.viz.plot_sensors(raw_task.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_task.info, show_names=True, kind='topomap')

mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='3d')
mne.viz.plot_sensors(raw_emptyroom.info, show_names=True, kind='topomap')

# =============================================================================
# PLOT 3D MONTAGE WITH SENSOR ORIENTATIONS (use task data only for the example)
# =============================================================================

fig = mne.viz.plot_sensors(raw_task.info, kind='3d', show_names=False)
ax = fig.gca()  # get 3D axis

positions = []
x_orient = []
y_orient = []
z_orient = []

for ch in raw_task.info['chs']:
    if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH:
        loc = ch['loc']
        positions.append(loc[:3])
        x_orient.append(loc[3:6])
        y_orient.append(loc[6:9])
        z_orient.append(loc[9:12])

positions = np.array(positions)
x_orient = np.array(x_orient)
y_orient = np.array(y_orient)
z_orient = np.array(z_orient)

# Quivers for each axis
scale = 0.02
ax.quiver(positions[:,0], positions[:,1], positions[:,2],
           x_orient[:,0], x_orient[:,1], x_orient[:,2],
           length=scale, color='red', normalize=True, label='X-axis')

ax.quiver(positions[:,0], positions[:,1], positions[:,2],
           y_orient[:,0], y_orient[:,1], y_orient[:,2],
           length=scale, color='green', normalize=True, label='Y-axis')

ax.quiver(positions[:,0], positions[:,1], positions[:,2],
          z_orient[:,0], z_orient[:,1], z_orient[:,2],
          length=scale, color='blue', normalize=True, label='Z-axis')

ax.legend()
plt.show()

plotname = 'plot_3D_w_orientations_montage'
name = "%s_%s" % (plotname, subj)
savepath = os.path.join(path_montages,name)
print(savepath)
plt.title(subj)
plt.savefig(savepath)

# =============================================================================
# SAVE MONTAGES (2D, 3D) [note: 3D w orientations was saved above]
# =============================================================================

mne.viz.plot_sensors(raw_task.info, show_names=True)
plotname = 'plot_montage'
name = "%s_%s" % (plotname, subj)
savepath = os.path.join(path_montages,name)
print(savepath)
plt.title(subj)
plt.savefig(savepath)

mne.viz.plot_sensors(raw_task.info, show_names=True, kind = '3d')
plotname = 'plot_3D_montage'
name = "%s_%s" % (plotname, subj)
savepath = os.path.join(path_montages,name)
print(savepath)
plt.title(subj)
plt.savefig(savepath)

# =============================================================================
# SAVE FIF FILES
# =============================================================================

raw_task.save(save_task, overwrite=True)
raw_emptyroom.save(save_emptyroom, overwrite=True)
     
# Print confirmation message with both filename and path
print(f"\nFile '{save_task}' successfully saved to:\n{save_task}.")
print(f"\nFile '{save_emptyroom}' successfully saved to:\n{save_emptyroom}.")
