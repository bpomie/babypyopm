#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
np.alltrue = np.all
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import mne
import os
import pandas as pd
import json

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

import utils_preprocessing_analysis
preprocess_analyse = utils_preprocessing_analysis.OPM_Pipeline(incl_report=False)

from matplotlib.collections import LineCollection
from io import BytesIO

# =============================================================================
# Parameters 
# =============================================================================
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

preprocessing_routine_input = 'processed_3_filter_ica_manualclean'

if preprocessing_routine_input == 'processed_1_filter':
    preprocessing_routine_output = 'preprocessing_routine_1'
elif preprocessing_routine_input == 'processed_2_filter_ica':
    preprocessing_routine_output = 'preprocessing_routine_2'
elif preprocessing_routine_input == 'processed_3_filter_ica_manualclean':
    preprocessing_routine_output = 'preprocessing_routine_3'
else:
    raise ValueError('Unknown preprocessing_routine_input')

path_results_rms = os.path.join(root_data_path, 'results',
                                preprocessing_routine_output, 'erf_grandaverage')
os.makedirs(path_results_rms, exist_ok=True)

path_manuscript_figures = os.path.join(root_data_path, 'results', 'manuscript_figures')
os.makedirs(path_manuscript_figures, exist_ok=True)

# =============================================================================
# Set up 
# =============================================================================
paths = utils.get_paths(root_data_path)

basename = f'{{subj}}/{preprocessing_routine_input}/{{sub-subj}}_evoked.fif'
dataset = study(os.path.join(paths.data, basename))

subjects = sorted([f for f in os.listdir(paths.data)
                   if os.path.isdir(os.path.join(paths.data, f))])
print(subjects)

all_subjects_evokeds = preprocess_analyse.load_evoked(
    dataset, subjects, paths.data, preprocessing_routine_input)

for subj_evokeds in all_subjects_evokeds:
    for evk in subj_evokeds:
        evk.pick_types(eeg=False, meg=True, misc=False)

# =============================================================================
# Grand-average across conditions, then across participants
# =============================================================================
per_subject_avg = [
    mne.combine_evoked([subj_evokeds[1], subj_evokeds[2]], weights='nave')
    for subj_evokeds in all_subjects_evokeds
]
grand_avg = mne.grand_average(per_subject_avg)

# =============================================================================
# Plot + save (panel B's approach: style joint plot, rasterize, embed)
# =============================================================================
topo_times = [-0.1, 0.075, 0.25, 0.425, 0.6]

# Output figure dimensions (cm)
FIG_WIDTH_CM   = 14.25
FIG_HEIGHT_CM  = 9.12
WIDTH_IN  = FIG_WIDTH_CM  / 2.54
HEIGHT_IN = FIG_HEIGHT_CM / 2.54

# Line widths (same as panel B)
ERF_LW       = 1.25
HEAD_LW      = 1.0
CONTOUR_LW   = 1.0
FRAME_LW     = 1.25

# Font sizes
AXIS_LABEL_SIZE      = 20
TICK_LABEL_SIZE      = 16
TITLE_SIZE           = 32
TOPO_TITLE_SIZE      = 12
CBAR_TICK_LABEL_SIZE = 14

# Colorbar cap
CBAR_VMAX = 437

# Colorbar surgery (panel B's relative approach)
CBAR_WIDTH_SCALE  = 0.1
CBAR_X_OFFSET     = 0.1

# Inner joint plot aspect: lower = more square = ts plot looks taller
INNER_WIDTH_SCALE = 0.7


def _find_colorbar(target_axis):
    cb = getattr(target_axis, '_colorbar', None)
    if cb is not None:
        return cb
    for ax in target_axis.figure.axes:
        for artist in list(ax.collections) + list(ax.images):
            cb = getattr(artist, 'colorbar', None)
            if cb is not None and cb.ax is target_axis:
                return cb
    return None


plt.close('all')
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['savefig.bbox'] = None
plt.rcParams['savefig.pad_inches'] = 0

# -----------------------------------------------------------------------------
# Step 1: render + style the joint plot
# -----------------------------------------------------------------------------
fig_joint = grand_avg.plot_joint(times=topo_times, show=False)

fig_joint.suptitle('')
for txt in fig_joint.texts:
    txt.set_text('')

ts_axis = max(fig_joint.axes,
              key=lambda a: a.get_position().width * a.get_position().height)
ts_axis.set_title('')
for txt in ts_axis.texts:
    txt.set_text('')

for axis in fig_joint.axes:
    for spine in axis.spines.values():
        spine.set_linewidth(FRAME_LW)

for axis in fig_joint.axes:
    axis.tick_params(axis='both', which='major',
                     labelsize=TICK_LABEL_SIZE, width=1, length=8)
    xl, yl = axis.get_xlabel(), axis.get_ylabel()
    if xl:
        axis.set_xlabel(xl, fontsize=AXIS_LABEL_SIZE)
    if yl:
        axis.set_ylabel(yl, fontsize=AXIS_LABEL_SIZE, labelpad=-32)
    if axis.get_title():
        axis.title.set_fontsize(TOPO_TITLE_SIZE)

for axis in fig_joint.axes:
    if axis is ts_axis:
        continue
    for line in axis.get_lines():
        line.set_linewidth(HEAD_LW)
    for coll in axis.collections:
        if isinstance(coll, LineCollection):
            coll.set_linewidth(CONTOUR_LW)

for line in ts_axis.get_lines():
    line.set_linewidth(ERF_LW)

ymin, ymax = ts_axis.get_ylim()
ts_axis.set_yticks(sorted(set([ymin, 0.0, ymax])))

xlim = ts_axis.get_xlim()
x_ticks = np.round(np.arange(xlim[0], xlim[1] + 1e-6, 0.1), 2)
ts_axis.set_xticks(x_ticks)

for axis in list(fig_joint.axes):
    if axis is ts_axis:
        continue
    cb = _find_colorbar(axis)
    if cb is None:
        continue

    vmin, _ = cb.mappable.get_clim()
    cb.mappable.set_clim(vmin, CBAR_VMAX)

    pos = cb.ax.get_position()
    mappable = cb.mappable
    orientation = getattr(cb, 'orientation', 'vertical')
    cb.remove()

    new_w  = pos.width * CBAR_WIDTH_SCALE
    new_x0 = pos.x0 + (pos.width - new_w) / 2 + CBAR_X_OFFSET
    new_ax = fig_joint.add_axes([new_x0, pos.y0, new_w, pos.height])
    new_cb = fig_joint.colorbar(mappable, cax=new_ax, orientation=orientation)

    ticks = [vmin, 0.0, CBAR_VMAX]
    new_cb.set_ticks(ticks)
    new_cb.set_ticklabels([f'{round(t)}' for t in ticks])
    new_ax.tick_params(axis='both', which='major',
                       labelsize=CBAR_TICK_LABEL_SIZE, width=1, length=8)

    break

w, h = fig_joint.get_size_inches()
fig_joint.set_size_inches(w * INNER_WIDTH_SCALE, h)

fig_joint.canvas.draw()

# -----------------------------------------------------------------------------
# Step 2: rasterize with tight bbox (captures the offset colorbar)
# -----------------------------------------------------------------------------
buf = BytesIO()
fig_joint.savefig(buf, format='png', dpi=300,
                  bbox_inches='tight', pad_inches=0)
plt.close(fig_joint)

# -----------------------------------------------------------------------------
# Step 3: embed into an output figure of exact dimensions
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=300)
fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.0)

buf.seek(0)
ax.imshow(plt.imread(buf))
ax.axis('off')

fig.suptitle('A. Grand-average EFR across participants',
             x=0.03, ha='left', fontweight='bold', fontsize=14)

fig.set_size_inches(WIDTH_IN, HEIGHT_IN, forward=True)
fig.set_dpi(300)
fig.canvas.draw()

out_path = os.path.join(path_manuscript_figures, 'fig3_panelA.png')
fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0)
print('saved to:', os.path.abspath(out_path))

plt.show()

from PIL import Image
img = Image.open(out_path)
print('pixels:', img.size)
print('dpi:', img.info.get('dpi'))
print('cm:', img.size[0]/300*2.54, img.size[1]/300*2.54)