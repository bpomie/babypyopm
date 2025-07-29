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

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Inser the path to your project folder
root_data_path = '/Users/b.pomiechowska@bham.ac.uk/Documents/GitHub/babypyopm/'

# =============================================================================
# SELECT PARTICIPANT
# =============================================================================

# Enter which participant you would like to explore
subj = 'sub-101'

# =============================================================================
# PATHS
# =============================================================================

# Construct paths
path_data  = os.path.join(root_data_path,'data')
path_montages  = os.path.join(root_data_path,'montages')

# Path task
path_task_data_raw = os.path.join(path_data,subj,'raw_rotated_sensorlocations',f"{subj}_file-oddballTones_upright_wsensorlocations_raw.fif")

# =============================================================================
# LOAD DATA (raw & filtered)
# =============================================================================

raw = mne.io.read_raw_fif(path_task_data_raw, preload=True)


#
# =============================================================================
# PLOT 3D MONTAGE WITH SENSOR ORIENTATIONS (use task data only for the example)
# =============================================================================

fig = mne.viz.plot_sensors(raw.info, kind='3d', show_names=False)
ax = fig.gca()  # get 3D axis

positions = []
x_orient = []
y_orient = []
z_orient = []

for ch in raw.info['chs']:
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

plotname = 'plot_3D_w_orientations_montage_all_axes'
name = "%s_%s" % (plotname, subj)
savepath = os.path.join(path_montages,name)
print(savepath)
plt.title(subj)
plt.savefig(savepath)

# =============================================================================
# PLOT 3D MONTAGE WITH SENSOR ORIENTATIONS (use task data only for the example)
# =============================================================================

fig = mne.viz.plot_sensors(raw.info, kind='3d', show_names=False)
ax = fig.gca()  # get 3D axis

positions = []
x_orient = []
y_orient = []
z_orient = []

for ch in raw.info['chs']:
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
# ax.quiver(positions[:,0], positions[:,1], positions[:,2],
#           x_orient[:,0], x_orient[:,1], x_orient[:,2],
#           length=scale, color='red', normalize=True, label='X-axis')

# ax.quiver(positions[:,0], positions[:,1], positions[:,2],
#           y_orient[:,0], y_orient[:,1], y_orient[:,2],
#           length=scale, color='green', normalize=True, label='Y-axis')

ax.quiver(positions[:,0], positions[:,1], positions[:,2],
          z_orient[:,0], z_orient[:,1], z_orient[:,2],
          length=scale, color='blue', normalize=True, label='Z-axis')

ax.legend()
plt.show()

plotname = 'plot_3D_w_orientations_montage_z_axis'
name = "%s_%s" % (plotname, subj)
savepath = os.path.join(path_montages,name)
print(savepath)
plt.title(subj)
plt.savefig(savepath)


