# babypyopm

# Preliminaries

#### Important information about the infant helmet / sensor layout

+ Sensor locations for the FieldLine infant rigid helmet are not integrated in the .fif file.
+ They are stored in a seperate .csv file `sensor_locations_mp_fixed.csv`.
+ Path: `~/data/project_micropiloting_infants_opm/sensor_locations_mp_fixed.csv`
+ Script `000_file_prep_infants.py` adds sensor locations to the **_raw.fif** files (see below for more details).

# Current data structure

<pre> ```
my_project/
├── index.html
├── style.css
└── scripts/
└── app.js
``` </pre>
    
```
project_setup_methods_paper
│   tbc.md
│   tbc.txt    
│
└───data
│   │
│   └───sub-001
│   │   │   file111.txt
│   │   │   file112.txt
│   │   │   ...
│
│
└───folder2
    │   file021.txt
    │   file022.txt
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
