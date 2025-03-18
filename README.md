# babypyopm

#### Important information about the infant helmet / sensor layout

+ The sensor locations for the infant helmet are not integrated in the .fif file.
+ They are stored in a seperate .csv file `sensor_locations_mp_fixed.csv`.
+ Path: `~/data/project_micropiloting_infants_opm/sensor_locations_mp_fixed.csv`
+ Script `000_file_prep_infants.py` adds sensor locations to the **_raw.fif** files (see below for more details).

> TO DO: store sensor locations for each participant in their data folder.

# Scripts

`000_file_prep_infants.py`
This script
