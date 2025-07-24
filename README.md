# babypyopm

# Preliminaries

#### Important information about the infant helmet / sensor locations

+ Sensor locations for the FieldLine infant rigid helmet are not integrated in the .fif file at recording.
+ They are stored in a seperate .tsv in each participants' folders, path: /data/sub-{subj}/📄 sub-{subj}_referencechannels_location.json
+ Script `file_prep_infants_add_sensor_locations.py` adds sensor locations to the **_raw.fif** files (see below for more details).

# Current data structure

```text
📁 project_setup_methods_paper
├── 📁 data
│   ├── 📁 sub-001
│   │   ├── 📁 raw_recording
│   │   ├── 📁 raw_rotated_sensorlocations
│   │   ├── 📁 processed_filtered
│   │   ├── 📁 processed_filtered_ica
│   │   ├── 📁 processed_filtered_{operation}
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
├── 📄 babyopm_testing_overview.csv
├── 💻 000_file_prep_infants_add_sensor_locations.py
├── 💻 001_simple_explore_psd_channels_noise.py
├── 💻 002_filtering.py
├── 💻 003_simple_explore_task.py utils_study
├── 📝 utils_study.py
├── 📝 utils_infant_helmet.py
├── 📝 utils_preprocessing_analysis.py
└── {...}
```
# Scripts

## Prep & channel checks

`000_file_prep_infants.py` - relevant only for the infant data

*input files*:\
*output file(s)*: *_upright_raw.fif

1. adds sensor locations
2. rotates the helmet (from supine to upright position)
3. concatenantes two tasks into one .fif file (background: tone oddball data and syllable oddball data were stored in seperate files)

> TO DO: pull (3) into a seperate script or embed in a if_else? This step is specific to the summer infant micropilots.

`001_file_prep_events.py`

*input files(*) infants: *_upright_raw.fif\
*input files(*) adults: *_raw.fif\
*output file(s)*: 

addes event descriptors from an event dictionary stored in the participant's folder /source_data/sub-* /sub-* _event_dict.json

`002_psd_channels.py`\
[currently a mess to be sorted]

`003_filtering.py`
1. filters the data
2. stores the output in `processed_data` folder

Therefore, the filtered (with standard parameters of your choice) data can be loaded for exploratory preprocessing / analysis steps without losing the time to run the filtering.

## Event related fields

`100_erfs`

## Decoding

### 500 Hz vs 750 Hz

### syllable 1 vs syllable 1 (ma vs te; bi vs go)

### Standard vs Odd

## Source localization

## Movement tracking
