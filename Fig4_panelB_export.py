#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 panel B
3x2 grid of per-subject RMS curves split by condition (frequent vs infrequent).

Data flow exactly mirrors 101_simple_plot_erf_rms.py:
    dataset = Study(...)                    # same basename template
    evoked_data = mne.read_evokeds(<path>)  # one path per subject
    mne.viz.plot_compare_evokeds({Frequent: evoked_data[1],
                                  Infrequent: evoked_data[2]},
                                 picks='mag', ci=None, ...)

OUTPUT
    fig4_panelB.png in results/manuscript_figures/
"""

import os

import numpy as np
np.alltrue = np.all

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mne

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# Arial first because macOS ships Helvetica as a .ttc collection that matplotlib
# can't unpack into separate weights, so Helvetica Bold doesn't render as bold.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# =============================================================================
# Parameters
# =============================================================================
root_data_path = '/Volumes/rdsprojects/o/oriolig-babyopm/project_setup_methods/'

PREPROCESSING_ROUTINE_MAP = {
    'processed_1_filter':                 'preprocessing_routine_1',
    'processed_2_filter_ica':             'preprocessing_routine_2',
    'processed_3_filter_ica_manualclean': 'preprocessing_routine_3',
}
preprocessing_routine_input = 'processed_3_filter_ica_manualclean'
if preprocessing_routine_input not in PREPROCESSING_ROUTINE_MAP:
    raise ValueError(f'Unknown preprocessing_routine_input: {preprocessing_routine_input}')
preprocessing_routine_output = PREPROCESSING_ROUTINE_MAP[preprocessing_routine_input]

paths = utils.get_paths(root_data_path)

path_manuscript_figures = os.path.join(root_data_path, 'results', 'manuscript_figures')
os.makedirs(path_manuscript_figures, exist_ok=True)

# Output dimensions (4.57" x 5.59" at 300 dpi)
WIDTH_IN  = 4.57
HEIGHT_IN = 5.59

# Conditions: (display label, evoked-list index, color)
# evoked_data[1] == frequent tones,  evoked_data[2] == infrequent tones
# (matching the comment block at top of 101_simple_plot_erf_rms.py)
CONDITIONS = [
    ('Frequent',   1, 'C0'),
    ('Infrequent', 2, 'C1'),
]
COLOR_DICT = {label: color for label, _, color in CONDITIONS}

# Plot styling (matched to Fig3 panels)
SUPTITLE_SIZE      = 18
SUBPLOT_TITLE_SIZE = 10
AXIS_LABEL_SIZE    = 9
TICK_LABEL_SIZE    = 8
LEGEND_SIZE        = 9
LINE_LW            = 1.25
SPINE_LW           = 0.75
TICK_WIDTH         = 0.6
TICK_LEN           = 3

# Baseline window (matches other figure 3 / 4 panels)
BASELINE = (0.0, 0.1)

# Fixed axis limits and tick positions
YLIM   = (0, 380)
YTICKS = [0, 100, 200, 300]
XTICKS = [0.0, 0.2, 0.4, 0.6]


# =============================================================================
# Discover evoked files using the SAME path template as 101_simple_plot_erf_rms.py
# =============================================================================
basename = f'{{subj}}/{preprocessing_routine_input}/{{sub-subj}}_evoked.fif'
dataset = study(os.path.join(paths.data, basename))

# dataset.match_files is the list of every evoked.fif under any subject folder.
# dataset.match_values[i] is {'subj': 'sub-XXX', 'sub-subj': 'sub-YYY'} for that file.
# Pair them and sort by subject id so the 3x2 grid fills in alphabetical order.
subject_files = sorted(
    [(mv['subj'], f) for mv, f in zip(dataset.match_values, dataset.match_files)],
    key=lambda x: x[0],
)

print(f'\nFound {len(subject_files)} evoked files:')
for subj, fpath in subject_files:
    print(f'  {subj}  ->  {fpath}')

if not subject_files:
    raise RuntimeError(
        f'No evoked files found under {paths.data}\n'
        f'Glob pattern was: {dataset.globdir}\n'
        f'Check that subject folders exist directly under {paths.data} '
        f'and contain {preprocessing_routine_input}/<subject>_evoked.fif'
    )


# =============================================================================
# Build figure
# =============================================================================
plt.close('all')
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = False

fig, axes = plt.subplots(3, 2, figsize=(WIDTH_IN, HEIGHT_IN), dpi=300,
                         constrained_layout=False)
axes_flat = axes.flatten()

# Cap at the first 6 subjects (grid is 3x2)
plot_pairs = subject_files[:len(axes_flat)]

# -----------------------------------------------------------------------------
# Pass 1: per subject, load evoked the same way the 101 script does, then let
# MNE draw the RMS curves into our subplot via axes=ax.
# -----------------------------------------------------------------------------
for ax, (subj, evoked_path) in zip(axes_flat, plot_pairs):
    # Identical to 101 script: just mne.read_evokeds, no wrapper, no pick_types.
    evoked_data = mne.read_evokeds(evoked_path)

    evokeds = {label: evoked_data[idx] for label, idx, _ in CONDITIONS}

    mne.viz.plot_compare_evokeds(
        evokeds,
        picks='mag',
        ci=None,
        colors=COLOR_DICT,
        axes=ax,
        legend=False,
        title='',
        show=False,
        truncate_yaxis=False,
        truncate_xaxis=False,
    )

    # Restyle the lines MNE just drew (mne default lw ~ 1.5)
    for line in ax.get_lines():
        line.set_linewidth(LINE_LW)

# Hide any unused axes (in case fewer than 6 subjects)
for i in range(len(plot_pairs), len(axes_flat)):
    axes_flat[i].set_visible(False)

# -----------------------------------------------------------------------------
# Pass 2: apply fixed ylim + tick positions + manuscript styling
# -----------------------------------------------------------------------------
for i, (ax, (subj, _)) in enumerate(zip(axes_flat, plot_pairs)):
    row, col = i // 2, i % 2
    is_bottom = (row == 2)
    is_left   = (col == 0)

    ax.set_ylim(YLIM)
    ax.set_yticks(YTICKS)
    ax.set_xticks(XTICKS)

    # Baseline shading + onset marker (matches other panels)
    ax.axvspan(*BASELINE, color='grey', alpha=0.2, zorder=0)
    ax.axvline(x=0, linestyle='--', color='black',
               linewidth=1.0, alpha=0.7, zorder=1)

    # Ticks + spines
    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_LABEL_SIZE, width=TICK_WIDTH, length=TICK_LEN)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    # Subject label is placed later via fig.text at each axis's tight-bbox
    # left edge, so left-column labels align with the y-axis label and
    # right-column labels align with the y-tick labels.

    # Axis labels only on outer subplots (overrides MNE's default ylabel)
    if is_bottom:
        ax.set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE, labelpad=2)
    else:
        ax.set_xlabel('')
    if is_left:
        ax.set_ylabel('RMS (fT)', fontsize=AXIS_LABEL_SIZE, labelpad=2)
    else:
        ax.set_ylabel('')

# Suptitle
fig.suptitle('B. Individual RMS split by condition',
             x=0.03, ha='left', fontweight='bold', fontsize=SUPTITLE_SIZE)

# Margins: leave room at the bottom for the legend
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.14,
                    wspace=0.30, hspace=0.45)

# Single horizontal legend at the bottom (one entry per condition)
legend_handles = [
    Line2D([0], [0], color=color, lw=LINE_LW, label=label)
    for label, _, color in CONDITIONS
]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(CONDITIONS),
           fontsize=LEGEND_SIZE, frameon=False,
           bbox_to_anchor=(0.5, 0.01))

fig.set_size_inches(WIDTH_IN, HEIGHT_IN, forward=True)
fig.set_dpi(300)
fig.canvas.draw()
print('post-set figsize:', fig.get_size_inches(), 'dpi:', fig.dpi)

# Place subject labels above each subplot, left-aligned to that subplot's
# tight bounding box. For left-column subplots this includes the "RMS (fT)"
# label, so the subject text starts where the y-axis label starts.
renderer = fig.canvas.get_renderer()
LABEL_Y_OFFSET = 0.005   # small vertical gap above each subplot, in figure coords
for ax, (subj, _) in zip(axes_flat, plot_pairs):
    bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0, bbox.y1 + LABEL_Y_OFFSET, subj,
             fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold',
             ha='left', va='bottom')

out_path = os.path.join(path_manuscript_figures, 'fig4_panelB.png')
fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0)
print('saved to:', os.path.abspath(out_path))

plt.show()

# Sanity check on output dimensions
from PIL import Image
img = Image.open(out_path)
print('pixels:', img.size)
print('dpi:', img.info.get('dpi'))
print('cm:', img.size[0]/300*2.54, img.size[1]/300*2.54)