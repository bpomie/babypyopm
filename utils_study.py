import re
import glob
import parse

import os
import csv
import mne
import logging
import json

from datetime import datetime

import pandas as pd

from string import Formatter
import numpy as np
np.alltrue = np.all
# from itertools import chain

import matplotlib.pyplot
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

from types import SimpleNamespace

class Study:
    """Class for simple file finding and looping.
    
    Parameters
    ----------
    studydir : str
        The study directory with wildcards.
    
    Attributes
    ----------
    studydir : str
        The study directory with wildcards.
    fieldnames : list
        The wildcards in the study directory, i.e., the field names in between {braces}.
    globdir : str
        The study directory with wildcards replaced with *.
    match_files : list
        The files that match the globdir.
    match_values : list
        The values of the field names (i.e., wildcards) for each file.
    fields : dict
        The field names and values for each file.
    
    Notes
    -----
    # Authors: 
    # Andrew Quinn <a.quinn@bham.ac.uk>
    # Mats van Es <mats.vanes@psych.ox.ac.uk>
    # https://github.com/OHBA-analysis/osl/blob/main/osl/utils/study.py
    
    This class is a simple wrapper around glob and parse. It works something like this:
    
    >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
    >>> study = Study(studydir)
    
    Get all files in the study directory:
    
    >>> study.get()
    
    Get all files for a particular subject:
    
    >>> study.get(subject='sub-01')
    
    Get all files for a particular subject and session:
    
    >>> study.get(subject='sub-01', session='ses-01')
    
    The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).
    """
    
    def __init__(self, studydir):
        """
        Notes
        -----
        This class is a simple wrapper around glob and parse. It works something like this:
        
        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)
        
        Get all files in the study directory:
        
        >>> study.get()
        
        Get all files for a particular subject:
        
        >>> study.get(subject='sub-01')
        
        Get all files for a particular subject and session:
        
        >>> study.get(subject='sub-01', session='ses-01')
        
        The fieldnames that are not specified in ``get`` are replaced with wildcards (*).
        """
        self.studydir = studydir

        # Extract field names in between {braces}
        self.fieldnames = [fname for _, fname, _, _ in Formatter().parse(self.studydir) if fname]
        print(studydir)
        print(self.fieldnames)

        # Replace braces with wildcards
        self.globdir = re.sub("\{.*?\}","*", studydir)
        print(self.globdir)

        self.match_files = sorted(glob.glob(self.globdir))
        print('found {} files'.format(len(self.match_files)))

        self.match_files = [ff for ff in self.match_files if parse.parse(self.studydir, ff) is not None]
        print('keeping {} consistent files'.format(len(self.match_files)))

        self.match_values = []
        for fname in self.match_files:
            self.match_values.append(parse.parse(self.studydir, fname).named)

        self.fields = {}
        # Use first file as a reference for keywords
        for key, value in self.match_values[0].items():
            self.fields[key] = [value]
            for d in self.match_values[1:]:
                self.fields[key].append(d[key])

    def get(self, check_exist=True, **kwargs):
        """Get files from the study directory that match the fieldnames.

        Parameters
        ----------
        check_exist : bool
            Whether to check if the files exist.
        **kwargs : dict
            The field names and values to match.

        Returns
        -------
        out : list
            The files that match the field names and values.

        Notes
        -----
        Example using ``Study`` and ``Study.get()``:
        
        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)
        
        Get all files in the study directory:
        
        >>> study.get()
        
        Get all files for a particular subject:
        
        >>> study.get(subject='sub-01')
        
        Get all files for a particular subject and session:
        
        >>> study.get(subject='sub-01', session='ses-01')
        
        The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).               
        """
        keywords = {}
        for key in self.fieldnames:
            keywords[key] = kwargs.get(key, '*')

        fname = self.studydir.format(**keywords)
        
        # we only want the valid files
        if check_exist:
            return [ff for ff in glob.glob(fname) if any(ff in ff_valid for ff_valid in self.match_files)]
        else:
            return glob.glob(fname)       

class Utils:

    def get_paths(self, root_data_path): #Barbara
        
        return SimpleNamespace(
            data=os.path.join(root_data_path, 'data'),
            data_recording=os.path.join(root_data_path, 'data/raw_recording'),
            data_rotated_sensorlocs=os.path.join(root_data_path, 'data/raw_rotated_sensorlocations'),
            data_filtered=os.path.join(root_data_path, 'data/processed_filtered'),
            results=os.path.join(root_data_path, 'results')
            )
    
    def load_concatenate_data(self, dataset, selected_subj, task_ids):
        
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        raws = []
        
        if selected_subj == 'all':
            subj_ids = [subject['subj_id'] for subject in self.subj_info['subjs'] 
                       if 'subj_id' in subject]
        else:
            subj_ids = selected_subj
        
        print(subj_ids)
        
        for sid in subj_ids:
            print(sid)
            
            taskfiles = []
            
            print(task_ids)
            
            for tid in task_ids:
                self.logger.info(f"Loading task data: {tid}")
                data_name = dataset.get(subj=sid, task=tid, extension='raw.fif')
    
                print(tid)
                print(data_name)            
    
                if len(data_name) == 0:
                    self.logger.warning(f"No file found for subject {sid}, task {tid}")
                    continue
            
                
                
                data_name = data_name[0]
                
                raw = mne.io.read_raw_fif(data_name, preload=True)
                raw_filename = raw.filenames[0]
                path, name = os.path.split(raw_filename)
                new_path = path.replace('/recording', '')
                filename, extension = os.path.splitext(name)
                filename = filename[:-14]
                raw.info['subject_info'] = {
                    'sid': sid,
                    'fif_path': new_path + '/',
                    'fif_name': filename
                }

                taskfiles.append(raw)
                data_ppt = mne.io.concatenate_raws(taskfiles, on_mismatch="raise")
                
            raws.append(data_ppt)
            
        return raws

    def mark_bad_channels(self, data, path):
        subid = self.get_participant_id(data)
        filepath = os.path.join(path, subid, subid + '_badchannels.tsv')
        print(path)
        print(filepath)
        known_bad_chns = self.read_csv(filepath)
        data.info["bads"].extend(known_bad_chns)
        return data
    
    def get_participant_id(self, raw): #Basia
        filename = raw.filenames[0]
        print(filename)
        subid = re.search(r'sub-\d+', filename).group()
        
        return subid
    
    def read_csv(self, filepath):
        tsvdata = pd.read_csv(filepath, sep='\t')
        tsvdatalist = list(tsvdata)
        print(tsvdatalist)
        return tsvdatalist
    
    def save_fif(self, raw, task, folder, fif_path, timestmp):
        
        sid = raw.info['subject_info']['sid'] # Get subject id
        
        new_filename = f"{sid}_file-{task}.fif"
        new_filename = new_filename.replace('/', '').replace('\\', '')
    
        new_full_path = os.path.join(fif_path,sid,folder,new_filename)
        raw.save(new_full_path, overwrite=True)
        
        # Print confirmation message with both filename and path
        print(f"\nFile '{new_filename}' successfully saved to:\n{new_full_path}")
        
        # Document in report
        #self.document_file_saving(new_filename, new_full_path, descriptor) 
        
    def save_fif_path(self, raw, task,descriptor, savepath,output_folder):
        
        filepath = raw.filenames[0]
        filename = os.path.basename(filepath)
        subid = self.get_participant_id(raw)
        datetime = filename.split('_')[0] + '_' + filename.split('_')[1]
        
        print(f"\n*** Procesing participant: {subid} ***")
        
        new_filename = f"{subid}_file-{task}_{descriptor}.fif"
        new_full_path = savepath + '/' + subid + '/' + output_folder + '/'+ new_filename
        
        raw.save(new_full_path, overwrite=True)
        
        # Document in report
        #self.document_file_saving(new_filename, new_full_path, descriptor) 
            
    def get_event_dict(self, raw, source_data_path):
        subid = self.get_participant_id(raw)
        subname = str(subid)
        print(subid)
        aux = os.path.join(source_data_path,subid)
        currpath = os.path.join(aux,subname+'_event_dict.json')
        
        with open(currpath, 'r') as f:
            event_dict = json.load(f)
        
        event_dict = {int(k): v for k, v in event_dict.items()}

        print(event_dict)
        
        return event_dict
    
    def annotate_events_breaks(self, raw, source_data_path, savepath): #Basia
        
        subid = self.get_participant_id(raw)
        subname = str(subid)
        
        event_dict = self.get_event_dict(raw, source_data_path)
        
        events = mne.find_events(raw)  # Annotate this line

        annot_from_events = mne.annotations_from_events(events=events,event_desc=event_dict,sfreq=raw.info["sfreq"]) # orig_time=raw.info["meas_date"], #TO DO: Double-check this
        raw.set_annotations(annot_from_events)

        break_annots = mne.preprocessing.annotate_break(raw, min_break_duration=1, t_start_after_previous=.25, t_stop_before_next=.25)
        raw.set_annotations(raw.annotations + break_annots)
        
        # Make plot of trigger events
        custom_mapping = {v: int(k) for k, v in event_dict.items()}
        mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=custom_mapping)
        plt.title(subname)
        plotname = 'plot_events'
        name = "%s_%s" % (plotname, subname)
        plt.savefig(savepath + name)    
        
        return raw