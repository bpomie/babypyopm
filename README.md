# babypyopm

#### Important information about the infant helmet / sensor layout

+ The sensor locations for the infant helmet are not integrated in the .fif file.
+ They are stored in a seperate .csv file `sensor_locations_mp_fixed.csv`.
+ Path: `~/data/project_micropiloting_infants_opm/sensor_locations_mp_fixed.csv`
+ Script `000_file_prep_infants.py` adds sensor locations to the **_raw.fif** files (see below for more details).

> TO DO: store sensor locations for each participant in their data folder.

# Scripts

`000_file_prep_infants.py` - relevant only for the infant data

*input files*:
/n *output file(s)*: *_upright_raw.fif

1. adds sensor locations
2. rotates the helmet (from supine to upright position)
3. concatenantes two tasks into one .fif file (background: tone oddball data and syllable oddball data were stored in seperate files)

> TO DO: pull (3) into a seperate script or embed in a if_else? This step is specific to the summer infant micropilots.

`001_file_prep_events.py`

*input files(*) infants: *_upright_raw.fif
*input files(*) adults: *_raw.fif
*output file(s)*: 

addes event descriptors from an event dictionary stored in the participant's folder /source_data/sub-*/sub-*_event_dict.json


