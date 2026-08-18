# babypyopm

# Repository info

This is a repository for Orioli*, G., Pesquita*, A., ..., Kowalczyk**, A., & Pomiechowska**, B. (under review). *Kick-starting infant OPM-MEG: proof-of-concept platform, open-science protocols, analysis pipeline BabyPy_OPM, and auditory oddball data.*

This repository comprises:
+ the infant OPM-MEG analysis pipeline **BabyPy_OPM**
+ Supplemental materials 01 to 12: https://github.com/bpomie/babypyopm/tree/main/supplemental_materials

The data, raw and processed, are available on the project OSF repository: https://osf.io/43xap/overview

# Preliminaries

#### Important information about the infant helmet / sensor locations

+ Sensor locations for the FieldLine prototype infant helmet are **not** integrated in the .fif file at recording.
+ They are stored in separate per-participant files (path: `~/data/sub-{nnn}/sub-{nnn}_sensor_locations.csv`) containing sensor positions (X, Y, Z, in mm) and orientation vectors (x_i ... z_k).
+ Script `000_simple_explore_layout_renamed_channels.py` adds sensor locations to the raw .fif files, renames the channels to slot-based labels, and applies the upright rotation (see below and the manuscript outline for more details).

#### Channel naming convention

After running script `000`, channels are named **side letter + slot number** (e.g. `L11`, `R04`), *without* the field-direction suffix (`_bz`). This matches the naming convention used in the `sub-{nnn}_badchannels.tsv` files, which downstream scripts rely on for bad-channel exclusion. Downstream scripts skip (with a warning) any bad-channel entries that do not match a channel in the recording — e.g. entries still using the pre-renaming convention (`s24_bz`); such entries should be updated to slot-style names via the mapping in `sub-{nnn}_sensor_locations.csv`.

#### Task design

The experimental task is an auditory oddball paradigm (`oddballTones`) with frequent and infrequent tones, each of which can be high or low pitch. Event codes are stored per participant in `sub-{nnn}_event_dict.json`, e.g.:

```json
{"infreq/tone/low": 2, "infreq/tone/high": 4, "freq/tone/low": 8, "freq/tone/high": 12}
```

MNE's hierarchical event naming allows selection by any level (e.g. `epochs["freq"]`, `epochs["infreq/tone"]`, `epochs["high"]`). Events are read from stim channel `di32`.

# Current project structure

```text
📁 project_setup_methods
├── 📁 data
│   ├── 📁 sub-101
│   │   ├── 📁 raw_recording
│   │   ├── 📁 raw_rotated_sensorlocations
│   │   ├── 📁 processed_1_filter
│   │   ├── 📁 processed_2_filter_ica
│   │   ├── 📁 processed_3_filter_ica_manualclean
│   │   ├── 📁 evoked
│   │   ├── 📄 sub-101_notes_session.txt
│   │   ├── 📄 sub-101_badchannels.tsv
│   │   ├── 📄 sub-101_sensor_locations.csv
│   │   ├── 📄 sub-101_event_dict.json
│   │   ├── 📄 sub-101_epochs_bad.txt
│   │   └── 📄 sub-101_referencechannels_location.json
│   ├── 📁 sub-102
│   ├── 📁 ...
│   └── 📁 sub-{nnn}
├── 📁 montages
├── 📁 results
│   ├── 📁 psd
│   ├── 📁 raw_amplitude
│   ├── 📁 preprocessing_routine_{1,2,3}
│   │   ├── 📁 erf
│   │   ├── 📁 rms
│   │   └── 📁 erf_grandaverage
│   ├── 📁 tfr_analysis
│   ├── 📁 decoding
│   └── 📁 manuscript_figures
├── 📄 participant_log.csv
├── 📄 babyopm_testing_overview.xlsx
├── 💻 000_simple_explore_layout_renamed_channels.py
├── 💻 001_simple_explore_psd_channels_noise.py
├── 💻 002_group_filtering.py
├── 💻 003_group_ica.py
├── 💻 004_group_manual_inspect.py
├── 💻 100_simple_explore_task.py
├── 💻 101_simple_plot_erf_rms.py
├── 💻 200_explore_grand_average.py
├── 💻 200_explore_grand_average_figure_export.py
├── 💻 300_time_frequency.py
├── 💻 400_temporal_decoding.py
├── 💻 500_group_splithalf_reliability.py
├── 📝 utils_study.py
└── 📝 utils_preprocessing_analysis.py
```

# Preprocessing routines

Three parallel preprocessing routines of increasing stringency are supported. Data folders hold the processed .fif files; the corresponding `results/preprocessing_routine_{n}` folders hold the outputs computed from them.

| Routine | Data folder | Processing |
|---|---|---|
| 1 | `processed_1_filter` | band-pass 0.1–40 Hz + notch |
| 2 | `processed_2_filter_ica` | routine 1 + ICA artifact removal |
| 3 | `processed_3_filter_ica_manualclean` | routine 2 + manual inspection / bad epoch marking |

Analysis scripts (`100`, `101`, `200`, `300`, `400`) take the routine as a parameter (`preprocessing_routine_input`, `preproc_folder`, or `input_folder`) so any analysis can be run on any routine.

# Scripts

**NOTE: Be sure to specify your working directory (`root_data_path`) at the top of each script before running.**

🔵 `simple_explore` scripts are single-participant walkthroughs using base MNE functions.

🔵 `group` scripts loop through all subjects in the data folder (or a manually specified subset) and use the `utils_*` pipeline modules.

## Files generated before the analysis

1. Files **generated automatically during recording**
(path: `~/data/sub-{nnn}/raw_recording/`)
  - `sub-{nnn}_channels.tsv`
  - `*_sub-{nnn}_file-oddballTones_raw.fif` (task recording)
  - `*_sub-{nnn}_file-emptyroom_raw.fif` (empty-room recording)

2. Files **generated during/after recording with manual experimenter input**
(path: `~/data/sub-{nnn}/`)
  - `sub-{nnn}_badchannels.tsv` — bad channel slots (column `badchannelslots`, slot-style names, e.g. `L11`)
  - `sub-{nnn}_sensor_locations.csv` — sensor positions and orientation vectors per channel
  - `sub-{nnn}_event_dict.json` — event code dictionary
  - `sub-{nnn}_referencechannels_location.json` — reference channel information
  - `sub-{nnn}_epochs_bad.txt` — bad epoch indices (one per line; produced during manual inspection, script `004`)

## Analysis scripts

### 1. `000_simple_explore_layout_renamed_channels.py`

Creates and applies sensor montages for OPM recordings from the per-participant sensor-location CSV. Processes both the task and the empty-room recording in one run. Works with any helmet layout provided as CSV (infant, adult, smart-helmet, custom).

Main steps:
- Load sensor geometry from `sub-{nnn}_sensor_locations.csv`
- Update sensor positions and orientation vectors in `raw.info` (channels absent from the CSV are zeroed; matching is reported)
- Drop MEG sensors left at position (0, 0, 0)
- Convert coordinates from mm to m
- Apply an optional rotation (`ROTATION`: `None`, `"z180"`, `"x180"`, or `"y180"`; `"z180"` brings the helmet upright)
- Rename channels to slot-based labels (`{side letter}{slot}`, e.g. `L11`; direction suffix dropped to match `badchannels.tsv`), with a duplicate-name check
- Build and apply an MNE montage
- Optionally anonymise the recording (`ANONYMISE = True` clears the measurement date)
- Save the updated FIF, montage plots, and a PSD check

+ ⬇️ **Inputs**
(path: `~/data/sub-{nnn}/raw_recording/` and `~/data/sub-{nnn}/`)
  - `*_sub-{nnn}_file-oddballTones_raw.fif`
  - `*_sub-{nnn}_file-emptyroom_raw.fif`
  - `sub-{nnn}_sensor_locations.csv`
+ 🛠️ **Outputs**
(path: `~/data/sub-{nnn}/raw_rotated_sensorlocations/`)
  - `sub-{nnn}_file-oddballTones_upright_wsensorlocations_raw.fif`
  - `sub-{nnn}_file-emptyroom_upright_wsensorlocations_raw.fif`
(path: `~/montages/`)
  - `plot_montage_{tag}_sub-{nnn}.png` (2D layout; tag = task name or `emptyroom`)
  - `plot_3D_montage_{tag}_sub-{nnn}.png` (3D layout)
  - `plot_3D_w_orientations_montage_{tag}_sub-{nnn}.png` (3D layout with X/Y/Z orientation vectors)

### 2. `001_simple_explore_psd_channels_noise.py`

Computes and saves Welch power spectral densities (0.1–125 Hz, n_fft = 10000) for the task and empty-room recordings, and plots raw sensor amplitudes ("lifts", in pT) before filtering.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/raw_rotated_sensorlocations/`)
  - `sub-{nnn}_file-oddballTones_upright_wsensorlocations_raw.fif`
  - `sub-{nnn}_file-emptyroom_upright_wsensorlocations_raw.fif`
+ 🛠️ **Outputs**
(path: `~/results/psd/`)
  - `sub-{nnn}_task.png`
  - `sub-{nnn}_emptyroom.png`
(path: `~/results/raw_amplitude/`)
  - `sub-{nnn}_raw_amplitude.png`

### 3. `002_group_filtering.py`

Group loop: band-pass filters (0.1–40 Hz) with notch filtering, via `utils_preprocessing_analysis.OPM_Pipeline.run_filter`.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/raw_rotated_sensorlocations/`)
  - `sub-{nnn}_file-oddballTones_upright_wsensorlocations_raw.fif`
+ 🛠️ **Output**
(path: `~/data/sub-{nnn}/processed_1_filter/`)
  - `sub-{nnn}_file-oddballTones_filtered_01_40.fif`

### 4. `003_group_ica.py`

Group loop: runs ICA on the filtered data via `OPM_Pipeline.run_ICA` and saves the cleaned recordings plus ICA diagnostics.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/processed_1_filter/`)
  - `sub-{nnn}_file-oddballTones_filtered_01_40.fif`
+ 🛠️ **Outputs**
(path: `~/data/sub-{nnn}/processed_2_filter_ica/`)
  - `sub-{nnn}_file-oddballTones_processed_2_filter_ica.fif`
(path: `~/results/processed_2_filter_ica/ica/`)
  - `sub-{nnn}_ICA_components.png`
  - `sub-{nnn}_ICA_excluded_component_{comp}.png`
  - `sub-{nnn}_ICA_sourcesA.png`, `sub-{nnn}_ICA_sourcesB.png`
(path: `~/results/logging/`)
  - `sub-{nnn}_excluded_ICA_components.tsv`

### 5. `004_group_manual_inspect.py`

Interactive visual inspection of the data (MNE browser, blocking). Bad segments/epochs identified by eye are recorded for later exclusion; the manually cleaned recording is saved.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/processed_2_filter_ica/`)
  - `sub-{nnn}_file-oddballTones_processed_2_filter_ica.fif`
+ 🛠️ **Outputs**
(path: `~/data/sub-{nnn}/processed_3_filter_ica_manualclean/`)
  - `sub-{nnn}_manual_clean.fif`
(path: `~/data/sub-{nnn}/`)
  - `sub-{nnn}_epochs_bad.txt` (bad epoch indices, used by scripts `400` and `500`)

### 6. `100_simple_explore_task.py`

Single-participant walkthrough of the task data for a selected preprocessing routine (`preprocessing_routine_input` parameter). Steps: set bad channels from `badchannels.tsv` (entries filtered to slot-style L/R names), PSD check, event extraction from stim channel `di32`, ISI computation and histogram, epoching (−0.1 to 0.6 s, detrend 1), ERF butterfly/joint/topomap plots, random equalization of epoch counts between conditions (seed 42), condition-wise ERFs, and RMS comparison of frequent vs infrequent. Saves the evoked responses for downstream group analysis.

+ ⬇️ **Inputs**
(path: `~/data/sub-{nnn}/{preprocessing folder}/`)
  - processed task .fif for the selected routine
(path: `~/data/sub-{nnn}/`)
  - `sub-{nnn}_badchannels.tsv`, `sub-{nnn}_event_dict.json`
+ 🛠️ **Outputs**
(path: `~/data/sub-{nnn}/evoked/`)
  - `sub-{nnn}_evoked.fif` (three evokeds: overall, frequent, infrequent; epoch counts equalized between conditions)
(path: `~/results/preprocessing_routine_{n}/erf/`)
  - `sub-{nnn}_erf_joint_overall.png`, `sub-{nnn}_erf_simple_overall.png`, `sub-{nnn}_erf_topo_overall.png`
  - `sub-{nnn}_erf_joint_freq.png`, `sub-{nnn}_erf_joint_infreq.png`
(path: `~/results/preprocessing_routine_{n}/rms/`)
  - `sub-{nnn}_rms.png`

### 7. `101_simple_plot_erf_rms.py`

Re-plots ERFs and the RMS comparison from previously saved evoked files, without recomputing epochs. Useful for quick figure iteration.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/{preprocessing folder}/`)
  - `sub-{nnn}_evoked.fif`
+ 🛠️ **Outputs**
(path: `~/results/preprocessing_routine_{n}/erf/`)
  - `sub-{nnn}_erf_joint_overall.png`, `sub-{nnn}_erf_joint_freq.png`, `sub-{nnn}_erf_joint_infreq.png`
(path: `~/results/preprocessing_routine_{n}/rms/`)
  - `sub-{nnn}_rms.png`

### 8. `200_explore_grand_average.py`

Loads all participants' evoked files for the selected preprocessing routine, restricts to MEG channels, groups by condition (overall / frequent / infrequent), and computes and plots grand averages.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/{preprocessing folder}/`)
  - `sub-{nnn}_evoked.fif` (all subjects)
+ 🛠️ **Outputs**
(path: `~/results/preprocessing_routine_{n}/erf_grandaverage/`)
  - `Overall_grand_average.png`, `Frequent_grand_average.png`, `Infrequent_grand_average.png`

### 9. `200_explore_grand_average_figure_export.py`

Manuscript figure export. Computes the grand averages (as in script `200`) and additionally builds a publication panel: per-participant condition-averaged joint plots (weighted combination of frequent and infrequent evokeds; topo times −0.1, 0.075, 0.25, 0.425, 0.6 s) assembled into a 3×2 grid at exact physical dimensions (14.35 × 14.65 cm, 300 dpi), with restyled fonts, line widths, colorbars, and axis ticks.

+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/{preprocessing folder}/`)
  - `sub-{nnn}_evoked.fif` (all subjects)
+ 🛠️ **Outputs**
(path: `~/results/preprocessing_routine_{n}/erf_grandaverage/`)
  - `Overall_grand_average.png`, `Frequent_grand_average.png`, `Infrequent_grand_average.png`
(path: `~/results/manuscript_figures/`)
  - `fig3_panelB.png` (individual ERFs across conditions, 3×2 grid)

### 10. `300_time_frequency.py`

Group time-frequency analysis. For each participant: sets bad channels (entries not matching any channel in the recording are skipped with a warning), epochs the data (−0.3 to 0.6 s, no baseline at epoching, detrend 1), equalizes epoch counts between conditions (seed 42), and computes averaged TFRs (10–40 Hz, n_cycles = 2) with both **multitaper** (time_bandwidth = 2.0) and **Morlet** methods. Plots use percent-change baseline correction (−0.1 to 0 s) over the −0.1 to 0.5 s window. Grand averages are computed and saved as reusable `.h5` files.

Note: with the 500 ms ISI of this paradigm, epoch windows are deliberately constrained, and TFR estimates are most robust at ≥ ~12 Hz.

+ ⬇️ **Inputs**
(path: `~/data/sub-{nnn}/{preproc_folder}/`; default `processed_1_filter`)
  - `sub-{nnn}_file-oddballTones_filtered_01_40.fif` (filename set by `preproc_naming`)
(path: `~/data/sub-{nnn}/`)
  - `sub-{nnn}_badchannels.tsv`, `sub-{nnn}_event_dict.json`
+ 🛠️ **Outputs**
(path: `~/results/tfr_analysis/{preproc_folder}/individual/`)
  - `sub-{nnn}_tfr_multitaper_10-40Hz_-0.1-0.5s.png`
  - `sub-{nnn}_tfr_morlet_10-40Hz_-0.1-0.5s.png`
  - `sub-{nnn}_tfr_analysis_info.txt`
(path: `~/results/tfr_analysis/{preproc_folder}/group_average/`)
  - `group_average_N{n}_multitaper_10-40Hz.png`, `group_average_N{n}_morlet_10-40Hz.png`
  - `group_multitaper_N{n}.h5`, `group_morlet_N{n}.h5`

### 11. `400_temporal_decoding.py`

Group temporal decoding (MVPA) with switchable contrast, discriminant pattern extraction, and a combined publication figure — all in one script.

**Rationale.** MVPA is well suited to infant OPM-MEG because sensor placement varies across participants (head sizes, positioning), which makes traditional univariate sensor-space comparisons difficult; the decoding approach requires no spatial alignment or sensor selection across participants.

**Workflow.** For each participant: set bad channels (non-matching entries skipped with a warning), epoch the data (−0.1 to 0.6 s, no baseline, detrend 1), optionally drop bad epochs from `sub-{nnn}_epochs_bad.txt` (`DROP_BAD_EPOCHS`, with out-of-range index guard), equalize epoch counts by random sampling (seed 42; minimum 20 epochs per condition), restrict to magnetometers with bad channels explicitly excluded, and run a **sliding estimator** (StandardScaler + logistic regression, liblinear) with 5-fold stratified cross-validation, scored by ROC AUC. The data needed for pattern extraction are cached so nothing is decoded twice.

At the group level: the peak of the mean post-stimulus decoding curve is identified with `scipy.signal.find_peaks`; a **binomial test** (chance = 0.5) evaluates how many participants decode above chance at the peak; an `mne.decoding.LinearModel` is fit at the group peak time per participant to recover interpretable **discriminant activation patterns**.

Contrast is selected via `DECODING_CONTRAST`:
- `'freq_vs_infreq'` — frequent vs infrequent tones
- `'high_vs_low'` — high vs low pitch tones

The final combined figure has two panels: **left**, Gaussian-smoothed (`SMOOTHING_SIGMA` = 10 samples, visualization only) temporal decoding curves per subject plus grand average ± SEM with peak annotation; **right**, per-subject discriminant activation pattern topographies at the group peak (3×2 grid with colorbars). All output paths and filenames carry the contrast name.

+ ⬇️ **Inputs**
(path: `~/data/sub-{nnn}/{input_folder}/`; input stage and filename set by `input_folder` and `fif_file_coda`)
  - processed task .fif
(path: `~/data/sub-{nnn}/`)
  - `sub-{nnn}_badchannels.tsv`, `sub-{nnn}_event_dict.json`, `sub-{nnn}_epochs_bad.txt` (optional)
+ 🛠️ **Outputs**
(path: `~/results/decoding/{input_folder}/{contrast}/decoding_individual/`)
  - `sub-{nnn}_temporal_decoding_{contrast}.png`
(path: `~/results/decoding/{input_folder}/{contrast}/decoding_group/`)
  - `group_temporal_decoding_{contrast}.png`
  - `group_decoding_results_{contrast}.csv` (time course: mean, SEM, per-subject AUC) and `.npy` (including peak time, peak AUC, binomial p)
  - `mvpa_results_summary_{contrast}.txt` (methods + results text for the paper)
  - `peak_subject_scores_{contrast}.csv` (per-subject AUC at group peak, above-chance flag)
(path: `~/results/decoding/{input_folder}/{contrast}/combined_figure/`)
  - `combined_decoding_figure_{contrast}.png` and `.pdf`

### 12. `500_group_splithalf_reliability.py`

Bootstrapped split-half reliability of the evoked response as a function of trial count. For each participant: sets bad channels, applies an extra 1–45 Hz IIR filter (5th-order Butterworth) to remove drift, epochs the data (−0.1 to 0.6 s), drops bad epochs from `sub-{nnn}_epochs_bad.txt`, applies baseline correction, then for subsample sizes of 10–390 trials (steps of 10) and 500 bootstrap iterations, splits the frequent-condition trials in half, correlates the mean evoked patterns (all channels × timepoints), and applies the Spearman–Brown correction. Runs in parallel via joblib.

Note: this script currently uses a hard-coded data directory (RDS project path) and subject list rather than the shared `root_data_path` convention.

+ ⬇️ **Inputs**
(path: `{datadir}/sub-{nnn}/processed_2_filter_ica/`)
  - `sub-{nnn}_file-oddballTones_processed_2_filter_ica.fif`
(path: `{datadir}/sub-{nnn}/`)
  - `sub-{nnn}_badchannels.tsv`, `sub-{nnn}_epochs_bad.txt`
+ 🛠️ **Outputs**
(path: working directory)
  - `reliability_figure.png` (uncorrected r vs number of trials, per subject)
  - `reliability_figure_withcorrection.png` (Spearman–Brown corrected)

# folder `montages`

Contains image files of sensor montages:
+ 📄 `*.png` 2D sensor layout plots
+ 📄 `*.png` 3D sensor layout plots
+ 📄 `*.png` 3D sensor layout plots with sensor orientation vectors

🛠️ Generated by: `000_simple_explore_layout_renamed_channels.py`

# folder `results`

## folder `psd`
Power spectral density (PSD) plots from **task** and **emptyroom** recordings.

🛠️ Generated by: `001_simple_explore_psd_channels_noise.py`

<!--**Examples**

<img src="results/psd/sub-102_emptyroom.png" width="2302" height="598" alt="sub-102 emptyroom PSD" />
<img src="results/psd/sub-102_task.png" width="2302" height="598" alt="sub-102 task PSD" />-->

## folder `raw_amplitude`
Raw sensor amplitude ("lifts") plots before filtering.

🛠️ Generated by: `001_simple_explore_psd_channels_noise.py`

## folders `preprocessing_routine_{1,2,3}`
Per-routine analysis outputs:
+ `erf/` — single-subject ERF plots (🛠️ `100_simple_explore_task.py`, `101_simple_plot_erf_rms.py`)
+ `rms/` — RMS comparison plots, frequent vs infrequent (🛠️ `100_simple_explore_task.py`, `101_simple_plot_erf_rms.py`)
+ `erf_grandaverage/` — grand average ERFs (🛠️ `200_explore_grand_average.py`, `200_explore_grand_average_figure_export.py`)

<!--**Example**

<img src="results/preprocessing_routine_1/rms/sub-102_rms.png" width="2302" height="598" alt="sub-102 RMS" />-->

## folder `tfr_analysis`
Time-frequency results, per preprocessing folder: `individual/` and `group_average/` subfolders with multitaper and Morlet TFR plots plus grand-average `.h5` files.

🛠️ Generated by: `300_time_frequency.py`

## folder `decoding`
Temporal decoding (MVPA) results, organized by preprocessing folder and contrast: `decoding_individual/`, `decoding_group/`, and `combined_figure/` subfolders.

🛠️ Generated by: `400_temporal_decoding.py`


