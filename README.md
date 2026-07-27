
# babypyopm

# Repository info

This is a repository for Orioli*, G., Pesquita*, A.,..., Kowalczyk**, A., & Pomiechowska**, B. (under review). Kick-starting infant OPM-MEG: proof-of-concept platform, open-science protocols, analysis pipeline BabyPy_OPM, and auditory oddball data.

This repository comprises:
+ infant OPM-MEG analysis pipeline _BabyPY_OPM_
+ Supplmental materials 01 to 12 [https://github.com/bpomie/babypyopm/tree/main/supplemental_materials]

The data, raw and processed, are available on the project OSF repository: https://osf.io/43xap/overview

# Preliminaries

#### Important information about the infant helmet / sensor locations

+ Sensor locations for the FieldLine prototype infant helmet are not integrated in the .fif file at recording.
+ They are stored in seperate .tsv files for each participant, path: `/data/sub-{subj}/📄 sub-{subj}_sensor_locations.json`
+ Script `000_simple_explore_layout.py` adds sensor locations to the **_raw.fif** files (see below and manuscript outline for more details).

# Current project structure

```text
📁 project_setup_methods_paper
├── 📁 data
│   ├── 📁 sub-001
│   │   ├── 📁 raw_recording
│   │   ├── 📁 raw_rotated_sensorlocations
│   │   ├── 📁 processed_1_filtered
│   │   ├── 📁 processed_2_filtered_ica
│   │   ├── 📁 processed_3_filtered_ica_manualclean
│   │   ├── 📄 sub-001_notes_session.txt
│   │   ├── 📄 sub-001_badchannels.tsv
│   │   ├── 📄 sub-001_sensor_locations.tsv
│   │   ├── 📄 sub-001_event_dict.json
│   │   └── 📄 sub-001_referencechannels_location.json
│   ├── 📁 sub-002
│   ├── 📁 ...
│   └── 📁 sub-{subj}
├── 📁 montages
├── 📁 results
│   ├── 📁 psd
│   ├── 📁 preprocessing_routine_1
│   │   ├── 📁 erf
│   │   └── 📁 rms
│   │   └── 📄 sub-001_referencechannels_location.json
│   ├── 📁 preprocessing_routine_2
│   ├── 📁 ...
│   └── 📁 preprocessing_routine_{routine}
├── ➡️ ➡️ 📄 participant_log.csv
├── 📄 babyopm_testing_overview.xlsx
├── 💻 000_simple_explore_layout.py
├── 💻 000_simple_explore_sensor_orientations.py
├── 💻 001_simple_explore_psd_channels_noise.py
├── 💻 002_group_filtering.py
├── 💻 003_group_ica.py
├── 💻 100_simple_explore_task.py 
├── 📝 utils_study.py
├── 📝 utils_preprocessing_analysis.py
└── {...}
```
# `scripts` and `files`

NOTE: Be sure to specify your working directory in each script.

:large_blue_circle: `simple_explore` scripts are walkthroughs on single participants and use base MNE functions. Please see manuscript outline for more details and tasks.

:large_blue_circle: `group` scripts loop through all subjects in the data folder. They use utils_* pipelines. Please see manuscript outline for more details and tasks.

## list of all `files` generated before the analysis

1. Files **generated automatically during recording**
(path: `~/data/sub-{nnn}/raw_recording/`)
-	`sub-{nnn}_channels.tsv`
-	`sub-{nnn}_raw.fif`

2. Files **generated during/after recording with manual experimenter input**
(path: '~/data/sub-{nnn}/')
  -	`sub-{nnn}_badchannels.tsv`
  -	`sub-{nnn}_sensor_locations.tsv`
  -	`sub-{nnn}_{sub-}_event_dict.json`
  -	`sub-{nnn}_referencechannels_location.json`

### list of all analysis `scrpts` and `files` generated during the analysis

1. Script **`000_simple_explore_layout_renamed_channels.py`**
[*function overview: to be completed*]
+ ⬇️ **Inputs**
(path: `~/data/sub-{nnn}/raw_recording/`)
  -	`sub-{nnn}_channels.tsv`
  -	`sub-{nnn}_raw.fif`
+ 🛠️ **Outputs**
(path: `~/data/sub-{nnn}/ raw_rotated_sensorlocations/`)
  -	`sub-{nnn}_upright_wsensorlocations_raw.fif`
(path: `~/montages/`)
  - [*to be completed*]

2. Script **`001_simple_explore_psd_channels_noise.py`**
[*function overview: to be completed*]
+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/ raw_rotated_sensorlocations/`)
  -	`sub-{nnn}_upright_wsensorlocations_raw.fif`
+ 🛠️ **Outputs**
(path: `~/results/psd/`)
  -	`{sub}_task.png`
  -	`{sub}_emptyroom.png`
(path: `~/results/lifts/)`
  -	`{sub}_before_filters.png`

3. Script **`002_group_filtering.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: `~/data/sub-{nnn}/raw_rotated_sensorlocations)`
  -	{name}_upright_wsensorlocations_raw.fif
+ Output
(path: `~/data/sub-{nnn}/ processed_1_filter)`
  -	`{name}_filtered_01_40.fif`

4. Script **`003_group_ica.py`**
[*function overview: to be completed*]
+ ⬇️ **Input**
(path: `~/data/sub-{nnn}/ processed_1_filter)`
  -	`sub-{nn}_filtered_01_40.fif'
+ **Outputs** 
(path: `~/data/sub-{nnn}/ processed_2_filter_ica`)
  -	`sub-{nnn}_filtered_01_40.fif`
(path: `~/results/processed_2_filter_ica/ica/`)
  -	`sub-{nnn}_ICA_components.png`
  -	`sub-{nnn}_ICA_excluded_component_{comp}.png`
  -	`sub-{nnn}_ICA_sourcesA.png`
  -	`sub-{nnn}_ICA_sourcesB.png`
(path: `~/results/logging/`)
  -	`sub-{nnn}_excluded_ICA_components.tsv`

5. Script **`004_group_manual_inspect.py`**
-	Identify bad epochs through visual inspection. 
-	Store bad epoch numbers for later use.
-	[*edit: store a summary about...*]
+ Input (path: `~/data/sub-{nnn}/ processed_2_filter_ica`)
  -	`sub-{nnn}_processed_2_filter_ica.fif`
+ Output (path: `~/data/sub-{nnn}/`)
  -	`sub-{nnn}_epochs_bad.csv`
  -	'sub-{nnn}_epochs_summary.csv`

6. Script **`100_simple_explore_task.py`**
[*function overview: to be completed*]
+ ⬇️ **Input**
(path: *to be completed*)
  - *to be completed*
+ **Outputs**
(path: *to be completed*)
  - *to be completed*

7. Script **`101_simple_plot_erf_rms.py`**
[*function overview: to be completed*]
+ ⬇️ **Input**
(path: *to be completed*)
  - *to be completed*
+ **Outputs**
(path: `*to be completed*`)
  - *to be completed*

8. Script **`200_explore_grand_average.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

9. Script **`200_explore_grand_average_figure_export.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

10. **Script `300_explore_tfr.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

11. **Script `300_simple_explore_tfr.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

12. **Script `300_simple_explore_tfr_ga.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

13. **Script `400_temporal_decoding_freq_vs_infreq.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

14. **Script `401_group_temporal_decoding.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

15. **Script `402_combined_decoding_figure.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

16. **Script `402_decoding_peak.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

17. **Script `402_group_decoding_peak.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

18. **Script `402a_group_decoding_compute.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

19. **Script `402b_group_decoding_figures.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

20. **Script `402c_group_decoding_patterns.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

21. **Script `500_group_splithalf_reliability.py`**
[*function overview: to be completed*]
+ ⬇️ Input
(path: *to be completed*)
  - *to be completed*
+ Outputs
(path: *to be completed*)
  - *to be completed*

# folder `montages`
contains image files of sensor motanges: 
  + 📄 `*.png` 2D sensor layout plots
  + 📄 `*.png` 3D sensor layout plots
  + + 📄 `*.png` 3D sensor layout plots with sensor orientations

🛠️ generated by: `000_file_prep_infants_add_sensor_locations.py`

# folder `results`

## folder `psd`
contains power spectral density (PSD) plots: 
  + 📄 `*.png` PSD plots from **task recordings**
  + 📄 `*.png` PSD plots from **emptyroom recordings**

🛠️ generated by: `001_simple_explore_psd_channels_noise.py`

**Examples**

<img src = "results/psd/sub-102_emptyroom.png" width="2302" height="598" alt="sub-102_erf_joint_overall" />
<img src = "results/psd/sub-102_task.png" width="2302" height="598" alt="sub-102_erf_joint_overall" />

## folder `rms`
contains rms plots

🛠️ generated by: `003_simple_explore_task.py`

**Examples**
<img src = "results/preprocessing_routine_1/rms/sub-102_rms.png" width="2302" height="598" alt="sub-102_erf_joint_overall" />

##folder [*to be continued*]






