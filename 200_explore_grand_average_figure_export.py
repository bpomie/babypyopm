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

# =============================================================================
# Parameters 
# =============================================================================
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SELECT PREPROCESSING ROUTINE
# =============================================================================
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
print(path_results_rms)
os.makedirs(path_results_rms, exist_ok=True)

# Manuscript figure output folder
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

conditions_grouped = list(zip(*all_subjects_evokeds))
grand_averages = [mne.grand_average(cond_list) for cond_list in conditions_grouped]

titles = ['Overall', 'Frequent', 'Infrequent']

for i, evk in enumerate(grand_averages, start=0):
    fig = evk.plot_joint()
    fig.suptitle(titles[i], fontsize=14)
    plt.show()
    fig.savefig(os.path.join(path_results_rms, titles[i] + '_grand_average'),
                dpi=300, bbox_inches="tight")

# =============================================================================
# Per-participant average across conditions — joint plots in a 2x3 grid
# =============================================================================
from io import BytesIO
from matplotlib.collections import LineCollection
import time

topo_times = [-0.1, 0.075, 0.25, 0.425, 0.6]

per_subject_avg = [
    mne.combine_evoked([subj_evokeds[1], subj_evokeds[2]], weights='nave')
    for subj_evokeds in all_subjects_evokeds
]

WIDTH_IN  = 14.35 / 2.54
HEIGHT_IN = 14.65 / 2.54

# Line widths
ERF_LW            = 1.25
HEAD_LW           = 1.0
CONTOUR_LW        = 1.0

# Element sizes
TOPOMAP_SCALE     = 1.0
INSET_SCALE       = 1.8
CBAR_WIDTH_SCALE  = 0.1
CBAR_X_OFFSET     = 0.1

# Font sizes
AXIS_LABEL_SIZE      = 22
TICK_LABEL_SIZE      = 18
TITLE_SIZE           = 36
CBAR_TICK_LABEL_SIZE = 16


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

fig, axes = plt.subplots(3, 2, figsize=(WIDTH_IN, HEIGHT_IN), dpi=300,
                         constrained_layout=False)
axes = axes.flatten()

for ax, subj, evk in zip(axes, subjects, per_subject_avg):
    fig_joint = evk.plot_joint(times=topo_times, show=False)

    fig_joint.suptitle('')
    for txt in fig_joint.texts:
        txt.set_text('')

    ts_axis = max(fig_joint.axes,
                  key=lambda a: a.get_position().width * a.get_position().height)
    ts_pos = ts_axis.get_position()

    ts_axis.set_title('')
    for txt in ts_axis.texts:
        txt.set_text('')

    def _is_inside(child, parent, tol=1e-3):
        return (child.x0 >= parent.x0 - tol and
                child.x1 <= parent.x1 + tol and
                child.y0 >= parent.y0 - tol and
                child.y1 <= parent.y1 + tol)

    cbars_to_resize = []
    for axis in fig_joint.axes:
        if axis is ts_axis:
            continue
        cb = _find_colorbar(axis)
        if cb is not None:
            cbars_to_resize.append((axis, cb))

    for axis in fig_joint.axes:
        axis.tick_params(axis='both', which='major',
                         labelsize=TICK_LABEL_SIZE, width=1, length=8)
        xl, yl = axis.get_xlabel(), axis.get_ylabel()
        if xl:
            axis.set_xlabel(xl, fontsize=AXIS_LABEL_SIZE)
        if yl:
            axis.set_ylabel(yl, fontsize=AXIS_LABEL_SIZE, labelpad=-38)
        if axis.get_title():
            axis.title.set_fontsize(TITLE_SIZE)
        for spine in axis.spines.values():
            spine.set_linewidth(1.25)

    cbar_axes_set = {a for a, _ in cbars_to_resize}
    for axis in fig_joint.axes:
        if axis is ts_axis or axis in cbar_axes_set:
            continue
        pos = axis.get_position()
        aspect = pos.width / pos.height if pos.height else 0
        is_square = 0.5 < aspect < 2.0
        is_inset  = _is_inside(pos, ts_pos) and is_square

        if is_square and not is_inset:
            axis.set_title('')

        for line in axis.get_lines():
            line.set_linewidth(HEAD_LW)
        for coll in axis.collections:
            if isinstance(coll, LineCollection):
                coll.set_linewidth(CONTOUR_LW)

        scale = INSET_SCALE if is_inset else (TOPOMAP_SCALE if is_square else 1.0)
        if scale != 1.0:
            cx = pos.x0 + pos.width  / 2
            cy = pos.y0 + pos.height / 2
            new_w = pos.width  * scale
            new_h = pos.height * scale
            axis.set_position(
                [cx - new_w / 2, cy - new_h / 2, new_w, new_h]
            )

    for line in ts_axis.get_lines():
        line.set_linewidth(ERF_LW)

    for old_axis, old_cb in cbars_to_resize:
        pos = old_axis.get_position()
        mappable = old_cb.mappable
        orientation = getattr(old_cb, 'orientation', 'vertical')

        old_cb.remove()

        new_w  = pos.width * CBAR_WIDTH_SCALE
        new_x0 = pos.x0 + (pos.width - new_w) / 2 + CBAR_X_OFFSET
        new_ax = fig_joint.add_axes([new_x0, pos.y0, new_w, pos.height])
        new_cb = fig_joint.colorbar(mappable, cax=new_ax, orientation=orientation)

        vmin, vmax = mappable.get_clim()
        ticks = [vmin, (vmin + vmax) / 2, vmax]
        new_cb.set_ticks(ticks)
        new_cb.set_ticklabels([f'{round(t)}' for t in ticks])
        new_ax.tick_params(axis='both', which='major',
                           labelsize=CBAR_TICK_LABEL_SIZE, width=1, length=8)

    w, h = fig_joint.get_size_inches()
    fig_joint.set_size_inches(w * 0.7, h)

    ymin, ymax = ts_axis.get_ylim()
    ts_axis.set_yticks(sorted(set([ymin, 0.0, ymax])))

    xlim = ts_axis.get_xlim()
    x_ticks = np.round(np.arange(xlim[0], xlim[1] + 1e-6, 0.1), 2)
    ts_axis.set_xticks(x_ticks)

    fig_joint.canvas.draw()

    buf = BytesIO()
    fig_joint.savefig(buf, format='png', dpi=150,
                      bbox_inches='tight', pad_inches=0)

    for n in plt.get_fignums():
        if n != fig.number:
            plt.close(n)

    buf.seek(0)
    ax.imshow(plt.imread(buf))
    ax.set_title(subj, fontsize=9, pad=6,
                 loc='left', fontweight='bold', x=0.03)
    ax.axis('off')

fig.suptitle('B. Individual EFRs accross conditions',
             x=0.03, ha='left', fontweight='bold')

fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.005,
                    wspace=0.0, hspace=0.15)

fig.set_size_inches(WIDTH_IN, HEIGHT_IN, forward=True)
fig.set_dpi(300)
fig.canvas.draw()
print('post-set figsize:', fig.get_size_inches(), 'dpi:', fig.dpi)

out_path = os.path.join(path_manuscript_figures, 'fig3_panelB.png')
fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0)
print('saved to:', os.path.abspath(out_path))

plt.show()

from PIL import Image
img = Image.open(out_path)
print('pixels:', img.size)
print('dpi:', img.info.get('dpi'))
print('cm:', img.size[0]/300*2.54, img.size[1]/300*2.54)