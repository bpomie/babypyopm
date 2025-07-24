import pandas as pd
import numpy as np
np.alltrue = np.all

import mne
import os
import json

import utils_study
utils = utils_study.Utils()
    
class SensorLayout:
    """Class for handling sensor layout operations in OPM data.
    
    Methods include rotation transformations, coordinate conversions, and sensor
    location management for both infant and reference helmets.
    
    Notes
    -----
    Authors: 
    - Ana Pesquita <a.pesquita@bham.ac.uk>
    - Anna Kowalczyk <a.u.kowalczyk@bham.ac.uk>
    - Barbara Pomiechowska <b.pomiechowska@bham.ac.uk>
    """
   
    def rotate_csv_z180_x90(self, df):
        """Rotate sensor positions by 180 degrees around Z-axis and 90 degrees around X-axis.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing sensor position data with columns for coordinates
            and orientation vectors.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rotated sensor positions.

        Notes
        -----
        Applies combined rotation matrices to coordinate groups:
        - Position coordinates: ['X', 'Y', 'Z']
        - X orientation vectors: ['x_i', 'x_j', 'x_k']
        - Y orientation vectors: ['y_i', 'y_j', 'y_k']
        - Z orientation vectors: ['z_i', 'z_j', 'z_k']
        """
        # Load the CSV file

        # Rotation matrix for 180 degrees around Z-axis
        rotation_matrix_z = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        # Rotation matrix for 90 degrees around X-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        # Combine rotation matrices
        combined_rotation = np.dot(rotation_matrix_x, rotation_matrix_z)

        # List of column groups to rotate
        column_groups = [
            ['X', 'Y', 'Z'],
            ['x_i', 'x_j', 'x_k'],
            ['y_i', 'y_j', 'y_k'],
            ['z_i', 'z_j', 'z_k']
        ]

        # Apply rotation to each group of columns
        for group in column_groups:
            rotated_coords = np.dot(df[group], combined_rotation)
            df[group] = rotated_coords

        return df
    
    def rotate_csv_z180(self, df):
        """Rotate sensor positions by 180 degrees around Z-axis.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing sensor position data with columns for coordinates
            and orientation vectors.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rotated sensor positions.

        Notes
        -----
        Applies rotation matrix to coordinate groups:
        - Position coordinates: ['X', 'Y', 'Z']
        - X orientation vectors: ['x_i', 'x_j', 'x_k']
        - Y orientation vectors: ['y_i', 'y_j', 'y_k']
        - Z orientation vectors: ['z_i', 'z_j', 'z_k']
        """
    
        # Rotation matrix for 180 degrees around Z-axis
        rotation_matrix_z = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    
    
        # List of column groups to rotate
        column_groups = [
            ['X', 'Y', 'Z'],
            ['x_i', 'x_j', 'x_k'],
            ['y_i', 'y_j', 'y_k'],
            ['z_i', 'z_j', 'z_k']
        ]
    
        # Apply rotation to each group of columns
        for group in column_groups:
            rotated_coords = np.dot(df[group], rotation_matrix_z)
            df[group] = rotated_coords
    
        return df

    def mm_to_m(self, df, column_names):
        """Transform values in specified DataFrame columns from millimeters to meters.
        
        Parameters
        ----------
        df : pandas.DataFrame 
            The input DataFrame containing measurement values
        column_names : list
            List of column names to transform

        Returns
        -------
        pandas.DataFrame
            DataFrame with specified columns converted from mm to m

        Raises
        ------
        ValueError
            If specified column is not found in DataFrame
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        for column_name in column_names:
            # Check if the column exists in the DataFrame
            if column_name not in df_copy.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in the DataFrame")

            # Convert the specified column from millimeters to meters
            df_copy[column_name] = df_copy[column_name] / 1000

            # Optionally, you can rename the column to indicate it's now in meters
            # df_copy.rename(columns={column_name: f"{column_name}_m"}, inplace=True)

        return df_copy
    
    # def read_infH_locs(self, csv_file, sid):
    #     """Read and process infant helmet sensor locations from CSV file.

    #     Parameters
    #     ----------
    #     csv_file : str
    #         Path to CSV file containing sensor locations
    #     sid : str
    #         Subject ID to determine which channel names to use

    #     Returns
    #     -------
    #     pandas.DataFrame
    #         Processed DataFrame containing sensor locations with appropriate channel names

    #     Notes
    #     -----
    #     Different processing is applied based on subject ID:
    #     - P02: Uses mp1_p02_channel_name
    #     - P03: Uses mp1_p03_channel_name
    #     - Others: Uses mp2_channel_name
    #     """
    #     df = pd.read_csv(csv_file)
        
    #     print(sid)

    #     if sid in ['sub-002']:
    #         print(sid)
    #         column_to_drop = ['mp1_p03_channel_name', 'mp2_channel_name'] 
    #         df = df.drop(columns=column_to_drop)
    #         df = df.dropna()
    #         df = df.rename(columns={'mp1_p02_channel_name': 'channel_name'})
    #         df = self.mm_to_m(df, ['X', 'Y', 'Z'])
    #     elif sid in ['sub-003']:
    #         column_to_drop = ['mp1_p02_channel_name', 'mp2_channel_name']
    #         df = df.drop(columns=column_to_drop)
    #         df = df.dropna()
    #         df = df.rename(columns={'mp1_p03_channel_name': 'channel_name'})
    #         df = self.mm_to_m(df, ['X', 'Y', 'Z'])
    #     elif sid in ['sub-004','sub-005','sub-006']:
    #         print('else')
    #         column_to_drop = ['mp1_p02_channel_name', 'mp1_p03_channel_name']
    #         df = df.drop(columns=column_to_drop)
    #         df = df.dropna()
    #         df = df.rename(columns={'mp2_channel_name': 'channel_name'})
    #         df = self.mm_to_m(df, ['X', 'Y', 'Z'])
    #     else:
    #         print('new new')

    #     return df
    
    def set_infH_loc(self, raw, df):
        """Assign coordinates from CSV template of infant helmet to raw data channel locations.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        df : pandas.DataFrame
            DataFrame containing channel location information

        Returns
        -------
        tuple
            raw : mne.io.Raw
                Raw data with updated channel locations
            inf_loc : dict
                Dictionary containing channel positions and orientation vectors

        Notes
        -----
        Updates channel locations with position and orientation vectors from CSV template
        """
        inf_loc = {}
        #dictionairy structure for the channel positions in the raw object
        for ch in raw.info['chs']:
            ch_name = ch['ch_name']

            # Store and print old position
            old_pos = ch['loc'][:3].copy()

            if ch_name in df['channel_name'].values:
                # Update the position based on CSV data
                pos = df[df['channel_name'] == ch_name][['X', 'Y', 'Z', 'x_i',
                                                         'x_j', 'x_k', 'y_i', 'y_j', 'y_k', 'z_i', 'z_j', 'z_k']].values[0]
                ch['loc'][:] = pos
                # print(f"Channel {ch_name} updated from CSV data.")

                # Extract sensor positions from raw_ref
                loc = ch['loc']
                pos = loc[:3]
                unit_vector_x = loc[3:6]
                unit_vector_y = loc[6:9]
                unit_vector_z = loc[9:12]
                inf_loc[ch_name] = {
                    'position': pos,
                    'unit_vector_x': unit_vector_x,
                    'unit_vector_y': unit_vector_y,
                    'unit_vector_z': unit_vector_z
                }

        return raw, inf_loc

    def get_refH_loc(self, referencepath):
        """Get reference helmet sensor locations from empty room recording.

        Parameters
        ----------
        referencepath : str
            Path to empty room recording file

        Returns
        -------
        dict
            Dictionary containing reference sensor positions and orientation vectors

        Notes
        -----
        - Renames adult helmet sensors to match infant helmet naming convention
        - Extracts position and orientation vectors for each channel
        """

        # infant helmet sensor names are of the following structure s*_bz
        # adult helmet sensor names are of the following structure R*_bz-s* or  L*_bz-s*
        # In this function I want to:
        # - rename the adult helmet sensors to follow the same format as the infant helmet sensors
        # - assigned adult helmet sensors to channel type ref_meg
        # The function inputs are raw_ref wich corresponds to an empty room recording with both infant and adult sensors "file-emptyandloc_raw.fif"

        raw_ref = mne.io.read_raw_fif(referencepath, preload=True)
        ref_loc = {}
        for ch in raw_ref.info['chs']:
            ch_name = ch['ch_name']
            if ch_name.startswith('s') and '_bz' in ch_name:
                raw_ref.drop_channels(ch_name)
            elif ch_name == 'di32':
                raw_ref.drop_channels(ch_name)
            else:

                # Rename the adult helmet channels
                # Extract the part after the last '-' and before '_bz' to rename the channel
                print(ch_name)
                try:
                    # Get the part after the last '-'
                    number_part = ch_name.split('-')[1]
                    if '_bz' in number_part:
                        # Extract the number part before '_bz'
                        number_part = number_part.split('_bz')[0]
                    # Ensure the number part doesn't already include 's' to avoid doubling 's'
                    if not number_part.startswith('s'):
                        new_name = f"s{number_part}_bz"
                    else:
                        new_name = f"{number_part}_bz"
                except IndexError:
                    # print(
                        # f"Warning: Couldn't process channel name {ch_name}. Skipping.")
                    continue

                # Extract sensor positions from raw_ref
                loc = ch['loc']
                pos = loc[:3]
                unit_vector_x = loc[3:6]
                unit_vector_y = loc[6:9]
                unit_vector_z = loc[9:12]
                ref_loc[new_name] = {
                    'position': pos,
                    'unit_vector_x': unit_vector_x,
                    'unit_vector_y': unit_vector_y,
                    'unit_vector_z': unit_vector_z
                }

        return ref_loc
    
    def set_refH_loc(self, raw, ref_loc, trns_vct):
        """Set reference helmet locations in raw data with translation.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        ref_loc : dict
            Dictionary containing reference sensor locations
        trns_vct : array-like
            Translation vector to apply to sensor positions

        Returns
        -------
        mne.io.Raw
            Raw data with updated reference sensor locations

        Notes
        -----
        - Updates channel locations with translated positions
        - Sets channel types to 'ref_meg' for reference channels
        """
        
        for ch in raw.info['chs']:
            ch_name = ch['ch_name']
          
            if ch_name in list(ref_loc.keys()):
                print('INSIDE set_refH_loc')
                print(ch_name)
                # Update the position based on reference data and apply translation
                # Get the correct ref position
                ref_position = np.array(ref_loc[ch_name]['position'])
                new_position = ref_position + trns_vct  # Apply translation
                
                # Update channel location in raw file
                ch['loc'][:3] = new_position
                ch['loc'][3:6] = ref_loc[ch_name]['unit_vector_x']
                ch['loc'][6:9] = ref_loc[ch_name]['unit_vector_y']
                ch['loc'][9:12] = ref_loc[ch_name]['unit_vector_z']
                
                # Set the channel types for reference channels
                ref_chn_dict = {ch_name: "ref_meg"}
                raw.set_channel_types(ref_chn_dict)

            # else:
            #    print(
            #        f"Warning: Channel {ch_name} not found in reference channels, skipping update.")

        return raw
    
    def add_sensor_locations(self, raw, dataset, task, source_data_path):

        subid = utils.get_participant_id(raw)
        subname = str(subid)
        aux = os.path.join(source_data_path,subname)
        currpath = os.path.join(aux,subname+'_referencechannels_location.json')
        print(currpath)
        sensor_locations_path = os.path.join(aux,subname+'_sensor_locations.csv')
        print(sensor_locations_path)
        
        print(subid)
        
        ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]

        # Check if there were reference sensors in the smart helmet
        smart_helmet_ref = self.check_if_smart_helmet_reference_sensors(ref_path)
        print('SMART HELMET REF')
        print(smart_helmet_ref)
        
        if task == 'emptyroom':
            
            if os.path.isfile(currpath):
                print("File exists")
                with open(currpath, 'r') as f:
                    ref_loc_files = json.load(f)
                
                channel_list_json = [name for name, info in ref_loc_files.items() if info['position'] == [0, 0, 0]]
                print(channel_list_json)
                # Build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
                type_map = {ch: 'ref_meg' for ch in channel_list_json}
                raw.set_channel_types(type_map)
                print(raw.info)
            
            if smart_helmet_ref == 1: #and task == 'emptyroom':
                ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]
                print('Ref_path is:')
                print(ref_path)
                raw_emptyroom = mne.io.read_raw_fif(ref_path, preload=True)
                channel_list = raw_emptyroom.ch_names
                channel_list_raw = raw.ch_names
                print('Print all channel names:')
                print(channel_list)
                print(channel_list_raw)
                smart_helmet_channels = [ch for ch in channel_list if ch.startswith(('R', 'L'))]
                print('Print all smart helmet channels:')
                print(smart_helmet_channels)
                # Build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
                type_map = {ch: 'ref_meg' for ch in smart_helmet_channels}
                raw.set_channel_types(type_map)
                print(raw.info)
                
        else:
            
            if smart_helmet_ref == 1:
                if os.path.isfile(currpath):
                    print("File exists")
                    with open(currpath, 'r') as f:
                        ref_loc_files = json.load(f)
                    ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]
                    ref_loc = self.get_refH_loc(ref_path)
                    print("Ref loc files")
                    print(ref_loc_files)
                    print("Ref loc smart helmet")
                    print(ref_loc)
                    print("refs from both sources")
                    ref_loc.update(ref_loc_files)
                else:
                    print("Only smart helmet")
                    ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]
                    ref_loc = self.get_refH_loc(ref_path)
            else:
                print("Only json")
                with open(currpath, 'r') as f:
                    ref_loc = json.load(f)
           
            # If infant micropilot arrange sensor layout 
            trns_vct = np.array([-0.55, 0.06, 0.15])
            raw = self.set_refH_loc(raw, ref_loc, trns_vct)
        
        # if smart_helmet_ref == 1:
        #     if os.path.isfile(currpath):
        #         print("File exists")
        #         with open(currpath, 'r') as f:
        #             ref_loc_files = json.load(f)
                
        #         channel_list_json = [name for name, info in ref_loc_files.items() if info['position'] == [0, 0, 0]]
        #         print(channel_list_json)
        #         # Build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
        #         type_map = {ch: 'ref_meg' for ch in channel_list_json}
        #         raw.set_channel_types(type_map)
        #         print(raw.info)
                
        #         #ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]
        #         #ref_loc = self.get_refH_loc(ref_path)
        #         print("Ref loc files")
        #         #print(ref_loc_files)
        #         print("Ref loc smart helmet")
        #         #print(ref_loc)
        #         print("refs from both sources")
        #         #ref_loc.update(ref_loc_files)
        #     else: # 23/7/25 Barbara
        #         print("Only smart helmet")
        #         ref_path = dataset.get(subj=subid, task='emptyroom', extension='raw.fif')[0]
        #         print('Ref_path is:')
        #         print(ref_path)
        #         raw_emptyroom = mne.io.read_raw_fif(ref_path, preload=True)
        #         channel_list = raw_emptyroom.ch_names
        #         print('Print all channel names:')
        #         print(channel_list)
        #         smart_helmet_channels = [ch for ch in channel_list if ch.startswith(('R', 'L'))]
        #         print('Print all smart helmet channels:')
        #         print(smart_helmet_channels)
        #         # Build mapping of smart helmet channels name to new type (i.e. 'ref_meg')
        #         type_map = {ch: 'ref_meg' for ch in smart_helmet_channels}
        #         raw.set_channel_types(type_map)
        #         #ref_loc = self.get_refH_loc(ref_path)

        # else:
        #     print("Only json")
        #     with open(currpath, 'r') as f:
        #         ref_loc = json.load(f)
            
        #with open('data.json', 'w') as json_file:
        #    json.dump(ref_loc, json_file)
                
            
        # #self.logger.info(f"Set infant helmet channel locations: {subid}") #EXPLAIN
        # #df = self.read_infH_locs(sensor_locations_path, subid)
        df = pd.read_csv(sensor_locations_path)
        df = self.mm_to_m(df, ['X', 'Y', 'Z'])
        print(df)
        df = self.rotate_csv_z180_x90(df)
        raw, inf_loc = self.set_infH_loc(raw, df)
       
        # # If infant micropilot arrange sensor layout 
        # trns_vct = np.array([-0.55, 0.06, 0.15])
        # #raw = self.set_refH_loc(raw, ref_loc, trns_vct)
        
        return raw
    
    def check_if_smart_helmet_reference_sensors(self, referencepath):
        
        raw_ref = mne.io.read_raw_fif(referencepath, preload=True)
        
        # Check if there is an element that does not start with "s" or "di"
        smart_channel = any(not (ch.startswith('s') or ch.startswith('di')) for ch in raw_ref.info['ch_names'])

        print(smart_channel)

        # Output 0 if not, output 1 if yes
        flag = 1 if smart_channel else 0

        print('REFERENCE SENSORS in SMART HELMET: [1: yes, 0: no]')

        print(flag)

        return flag
    
    def align_topo_locs(self, raw):
        
        # Get the current sensor positions
        pos = raw.info['chs']
        
        # Extract x, y, z coordinates
        coords = np.array([ch['loc'][:3] for ch in pos])
        
        # Create rotation matrix for 90 degree clockwise rotation around x-axis
        theta_x = np.radians(-90)  # Note the negative angle for clockwise rotation
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        
        # Apply rotation
        new_coords = np.dot(coords, rotation_matrix.T)
        
        # Update the channel locations in the raw object
        for i, ch in enumerate(raw.info['chs']):
            ch['loc'][:3] = new_coords[i]
        
        # Update the dev_head_t transformation
        if raw.info['dev_head_t'] is not None:
            dev_head_t = raw.info['dev_head_t']['trans']
            dev_head_t[:3, :3] = np.dot(rotation_matrix, dev_head_t[:3, :3])
            raw.info['dev_head_t']['trans'] = dev_head_t
        
        return raw



    
        
