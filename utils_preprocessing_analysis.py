import re
import glob
import parse

from string import Formatter
import pandas as pd
import numpy as np
np.alltrue = np.all
# from itertools import chain
import mne

import os
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg

import logging
import json
import csv
from pathlib import Path
from datetime import datetime

from matplotlib.gridspec import GridSpec

import glob
       
class OPM_Pipeline:
    """A class to handle MEG/EEG data preprocessing with automated reporting.
    
    Parameters
    ----------
    root_data_path : str or Path
        Path to the root directory containing the data and auxiliary files.
    incl_report : bool
        Whether to include automated report generation
    
    Attributes
    ----------
    root_path : Path
        Path object for root directory
    logger : logging.Logger
        Logger instance for the class
    subj_info : dict
        Subject information loaded from JSON file
    report : mne.Report
        MNE Report object for preprocessing documentation
    report_name : str
        Name of the report file
    report_path : Path
        Path to the HTML report file
        
    Notes
    -----
    Authors:
    - Ana Pesquita <a.pesquita@bham.ac.uk>
    - Andrew Quinn <a.quinn@bham.ac.uk>
    - Anna Kowalczyk <a.u.kowalczyk@bham.ac.uk>
    - Barbara Pomiechowska <b.pomiechowska@bham.ac.uk>
    """
    
    def __init__(self, incl_report):
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Convert to Path object and validate root path
        #self.root_path = Path(root_data_path)
        #self.loc_path = self.root_path / 'data' / 'project_micropiloting_infants_opm'
        
        # Initialize report attributes
        self.report = None
        self.report_name = None
        self.report_path = None 
        
        # Load subject information
        #self.subj_info = self._load_subject_info() #B
        
        if incl_report==True:
            # Initialize report
            self._initialize_report()
    
    def _initialize_report(self):
        """Initialize the MNE report with user input.

        Notes
        -----
        - Prompts user for report filename
        - Collects pipeline description
        - Creates and saves initial report
        """
        # Get report filename
        while True:
            self.report_name = input("Enter the filename for the report (e.g., 'report'): ").strip()
            if self.report_name:
                self.report_path = Path(self.report_name).with_suffix('.html')
                break
            self.logger.warning("Report filename cannot be empty")
        
        # Get report description
        print("\nEnter a short description of the pipeline.")
        print("Example: <p>Bandpass 0.1-150Hz and Line-notch at 50Hz.</p>")
        print("Press Enter twice when finished:\n")
        
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        self.html_content = "\n".join(lines)
        
        # Create and save report
        self.report = self.create_report()
        self.logger.info(f"Report initialized at {self.report_path}")
    
    def get_paths(self, sensorlocations): #Barbara
        paths = [row for row in csv.reader(open('paths.tsv', 'r'))]
        paths = {row[0].split('\t')[0]: row[0].split('\t')[1] for row in paths} # Convert to dictionary

        root_data_path = paths.get('root_data_path')
        path_source_data = os.path.join(root_data_path,'data/project_micropiloting_infants_opm/source_data')
        path_sensorlocations = os.path.join(root_data_path,'data/project_micropiloting_infants_opm',sensorlocations)
        path_results = os.path.join(root_data_path, 'results/')
        
        return path_source_data, path_sensorlocations, path_results

    
    def create_report(self):
        """Create and initialize the MNE Report object.
        
        Returns
        -------
        mne.Report
            Configured report object with initial content

        Notes
        -----
        - Creates report with metadata
        - Adds timestamp and pipeline description
        - Saves report to file
        """
        # Create report with metadata
        report = mne.Report(
            title='Infant OPM Pipeline',
            info_fname=self.report_path,
            subject=self.subj_info.get('subject_id')        )
        
        # Add timestamp and pipeline description
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = f"""
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Pipeline Description:</strong> </p>
        {self.html_content}
        """
        
        report.add_html(title='Pipeline Description', html=header)
        
        try:
            report.save(self.report_path, overwrite=True)
            self.logger.info(f"Report saved successfully to {self.report_path}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            raise
        
        return report
     
    def drop_bad_ch(self, raw, known_bads):
        """Mark and drop known bad sensors from raw data.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        known_bads : list
            List of known bad channel names

        Returns
        -------
        mne.io.Raw
            Raw data with bad channels dropped
        """
        # - mark known bad sensors as based
        # Ensure there are no duplicates before adding
        current_bads = set(raw.info['bads'])
        known_bads_set = set(known_bads)
        all_bads = current_bads.union(known_bads_set)
        raw.info['bads'] = list(all_bads)
        raw.drop_channels(raw.info['bads'])
        return raw
    
    def plot_montage(self, aux, path): #Basia
        raw = aux.copy()
        raw.drop_channels(['di32'])
        ch = raw.info['ch_names']
        print(ch)
        mne.viz.plot_sensors(raw.info, show_names=True)
        subid = self.get_participant_id(raw)
        subname = str(subid)
        plotname = 'plot_montage'
        name = "%s_%s" % (plotname, subname)
        #plt.show()
        savepath = os.path.join(path,name)
        print(savepath)
        plt.title(subname)
        plt.savefig(savepath)
        
    def plot_montage_3D(self, aux, path): #Basia
        raw = aux.copy()
        raw.drop_channels(['di32'])
        ch = raw.info['ch_names']
        print(ch)
        mne.viz.plot_sensors(raw.info, show_names=True, kind='3d')
        subid = self.get_participant_id(raw)
        subname = str(subid)
        plotname = 'plot_montage3D'
        name = "%s_%s" % (plotname, subname)
        #plt.show()
        savepath = os.path.join(path,name)
        print(savepath)
        plt.title(subname)
        plt.savefig(savepath)
    
    def get_participant_id(self, raw): #Basia
        filename = raw.filenames[0]
        subid = re.search(r'sub-\d+', filename.name).group()
        
        return subid
    
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
    
    def get_json_file(self, raw, source_data_path, filename):
        subid = self.get_participant_id(raw)
        subname = str(subid)
        print(subid)
        aux = os.path.join(source_data_path,subid)
        currpath = os.path.join(aux,subname+filename)
        
        with open(currpath, 'r') as f:
            aux = json.load(f)
        
        aux = {int(k): v for k, v in aux.items()}

        print(aux)
        
        return aux
    
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
        
    
    def load_data(self, dataset, selected_subj,pathdata,input_folder,taskname): #TO FIX
        
        alldata = []
        
        if selected_subj == 'all':
            subj_ids = [subject['subj_id'] for subject in self.subj_info['subjs'] 
                       if 'subj_id' in subject]
        else:
            subj_ids = selected_subj
            
        print(subj_ids)
        
        for sid in subj_ids:
            print(sid)            
            pattern = os.path.join(pathdata,sid,input_folder,f"{sid}_file-{taskname}_*.fif")
            match = glob.glob(pattern)
            data_name = match[0]
            print(data_name)
            
            data_ppt = mne.io.read_raw_fif(data_name, preload=True)
            alldata.append(data_ppt)
            
        return alldata
    
    def load_evoked(self, dataset, selected_subj,pathdata,input_folder): #TO FIX
        
        alldata = []
        
        if selected_subj == 'all':
            subj_ids = [subject['subj_id'] for subject in self.subj_info['subjs'] 
                       if 'subj_id' in subject]
        else:
            subj_ids = selected_subj
            
        print(subj_ids)
        
        for sid in subj_ids:
            print(sid)            
            pattern = os.path.join(pathdata,sid,input_folder,f"{sid}_evoked.fif")
            print(pattern)
            match = glob.glob(pattern)
            data_name = match[0]
            print(data_name)
            
            data_ppt = mne.read_evokeds(data_name)
            alldata.append(data_ppt)
            
        return alldata
    
    def load_concatenate_data(self, dataset, selected_subj, task_ids):
        
        raws = []
        
        if selected_subj == 'all':
            subj_ids = [subject['subj_id'] for subject in self.subj_info['subjs'] 
                       if 'subj_id' in subject]
        else:
            subj_ids = selected_subj
        
        for sid in subj_ids:
            print(sid)
            
            taskfiles = []
            
            for tid in task_ids:
                self.logger.info(f"Loading task data: {tid}")
                data_name = dataset.get(subj=sid, task=tid, extension='raw.fif')
    
                if len(data_name) == 0:
                    self.logger.warning(f"No file found for subject {sid}, task {tid}")
                    continue
                print(data_name)
                
                data_name = data_name[0]
                
                raw = mne.io.read_raw_fif(data_name, preload=True)
                raw_filename = raw.filenames[0]
                path, name = os.path.split(raw_filename)
                new_path = path.replace('/recordingfiles', '')
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
            
    #def intergrate_sensor_locations(self):
        
    
    def add_sensor_locations(self, raw, dataset, source_data_path, sensor_locations_path):
        
        # Load class with sensor layout functions
        sensorlayout = SensorLayout()

        subid = self.get_participant_id(raw)
        subname = str(subid)
        aux = os.path.join(source_data_path,subname)
        currpath = os.path.join(aux,subname+'_referencechannels_location.json')
        
        print(subid)
        
        if subid == 'sub-002' or subid == 'sub-003':
            with open(currpath, 'r') as f:
                ref_loc = json.load(f)
        else:
            ref_path = dataset.get(subj=subid, task='emptyandloc', extension='raw.fif')[0]
            print(ref_path)
            ref_loc = sensorlayout.get_refH_loc(ref_path)
            
        self.logger.info(f"Set infant helmet channel locations: {subid}")
        df = sensorlayout.read_infH_locs(sensor_locations_path, subid)
        print(df)
        df = sensorlayout.rotate_csv_z180_x90(df)
        raw, inf_loc = sensorlayout.set_infH_loc(raw, df)
       
        # If infant micropilot arrange sensor layout 
        trns_vct = np.array([-0.55, 0.06, 0.15])
        raw = sensorlayout.set_refH_loc(raw, ref_loc, trns_vct)
        
        return raw
    
    
    def run_filter(self, raw, l_freq, h_freq, notch):
        
        if notch == True:
            raw.notch_filter(freqs=np.arange(50, 251, 50), notch_widths=5)
        
        raw.filter(l_freq, h_freq)
        #raw.resample(sfreq=1000)
        
        return raw
    
    def plot_psd_adults(self, raw, plotmaxfreq):
        
        desired_sfreq = 1000
        #current_sfreq = raw.info['sfreq']

        #lowpass_freq = desired_sfreq / 4.0
        #raw_resampled = raw.copy().filter(l_freq=None, h_freq=lowpass_freq)

        raw_resampled = raw.copy()
        raw_resampled.resample(sfreq=desired_sfreq)

        n_fft = 10000
        raw_PSD = raw_resampled.compute_psd(method="welch", fmin=.1, fmax=plotmaxfreq, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
        raw_PSD.plot()
        
        return raw
    
    def run_epoching(self, data, tmin, tmax, detr, conditions, process, source_data_path):
        
        subid = self.get_participant_id(data)
        subname = str(subid)
        print(subname)
        
        event_dict = self.get_event_dict(data, source_data_path)
        mapping = {v: int(k) for k, v in event_dict.items()}
        events = mne.find_events(data)
        
        print(data.info["bads"])

        epochs = mne.Epochs(data, events, tmin=tmin, tmax=tmax, detrend=detr, event_id=mapping, reject_by_annotation=True)
        
        print(epochs)
        
        evoked = epochs.average()
        #fig = evoked.plot(exclude='bads',spatial_colors=True, gfp=True)
        fig = evoked.plot_joint()
        fig.suptitle(subname)
        
        print(epochs)
        
        return epochs
        
    def plot_something(self, epoched_data, subname, conditions, process, savepath, projectname):
        
        print("Epoched data:", epoched_data)
        #subid = self.get_participant_id(data)
        #subname = epoched_data[0][1]
        print('***')    
        print(subname)
        print('***')
        
        print(epoched_data)
        
        myfigs = [self.plot_butterfly(c, epoched_data, process, savepath, subname) for c in conditions]
        
        # Create a grid of subplots
        num_figs = len(myfigs)
        fig, axes = plt.subplots(1, num_figs, figsize=(25, 5))
        
        # Set the title for the entire figure
        figname = subname + '_' + process
        print(figname)
        fig.suptitle(figname)

        # Plot each figure in the grid
        for i, myfig in enumerate(myfigs):
            myfig.axes[0].figure = fig
            myfig.axes[0].set_position(axes[i].get_position())
            axes[i].imshow(myfig.canvas.buffer_rgba())
            # Add title to each subplot
            #axes[i].set_title(f"Title {i+1}")
            axes[i].set_title(conditions[i])
            # Remove axes markings
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            # Remove the frame (spines)
            for spine in axes[i].spines.values():
                spine.set_visible(False)


        plt.show()
        
        plt.savefig(savepath + projectname + '/' + subname + '_' + process, dpi=300)
        

    def plot_butterfly(self, c, data, process, savepath, subname):
        
        print(c)
        ts_args = dict(ylim=dict(mag=[-650, 650]))
        
        evoked = data[c].average()
        
        fig = evoked.plot_joint(exclude = 'bads', times=[0., 0.1, 0.2, 0.3, 0.4, 0.5], ts_args=ts_args)
        #evoked.plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4, 0.5])
        fig.suptitle(c)
        #fig.set_size_inches(10, 20)
        
        plotname = 'plot_butterfly'
        name = "%s_%s_%s_%s" % (plotname, subname, process, c)
        print(name)
        print(savepath)
        
        #plt.title('sdfsdfds')
        #plt.savefig(savepath + name, dpi=300)
        
        plt.show()
        plt.close(fig)
        
        return fig

    def plot_psd(self,data,n_fft,fmax,savepath):
        auxpsd = data.compute_psd(method="welch", fmax=fmax, picks="mag", n_fft=n_fft)
        fig = auxpsd.plot(spatial_colors=True)
        
        subid = self.get_participant_id(data)
        subname = str(subid)
        fmax = str(fmax)
        print(subname)
        plotname = 'plot_psd_raw'
        name = "%s_%s_%sfmax" % (plotname, subname, fmax)
        #plt.show()
        print(savepath)
        
        plt.title(name)
        plt.savefig(savepath + name, dpi=300)
        plt.close(fig)
        
        
        #save_path = os.path.join(results_path, 'psd.png')
        #print(save_path)
        #fig1.savefig('psd_plot_newtry.png')
        #fig1.savefig()
        
    
# =============================================================================
#     def run_filtering(self, raw, l_freq, h_freq, make_plot):
#         """Apply bandpass and notch filters to the raw data.
# 
#         Parameters
#         ----------
#         raw : mne.io.Raw
#             Raw instance to filter
#         l_freq : float
#             Lower frequency bound for bandpass filter
#         h_freq : float
#             Upper frequency bound for bandpass filter
#         make_plot : bool
#             Whether to create PSD plots before/after filtering
# 
#         Returns
#         -------
#         mne.io.Raw
#             Filtered raw data
# 
#         Notes
#         -----
#         - Applies bandpass filter from l_freq to h_freq
#         - Applies notch filter at line frequencies (50Hz and harmonics)
#         - Creates PSD plots if make_plot=True
#         - Saves filtered data to file
#         """
#         
#         mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs']), None)
#         if "adult" not in mp_nr:
#             print("Aligning infant helmet")
#             raw = self.align_topo_locs(raw)
#         raw_filename = raw.filenames[0]
#         match = re.search(r'(\d{8}_\d{6}_sub-P\d+)', raw_filename)
#         cur_pptdata = str(match.group(1))
#         sid=raw.info['subject_info']['sid']
#         print(f"Starting filtering on data: {sid}")
#         
#         if make_plot:
#             # Create figure first, then create subplots
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#             # Create first subplot for pre-filtering PSD
#             raw.plot_psd(show=False, ax=ax1, fmin=l_freq, fmax=h_freq, average=False)
#             plt.title('Before Filtering')
#             
#         print(f"Bandpass filter from {l_freq}Hz to {h_freq}Hz: {sid}")
#         raw.filter(l_freq, h_freq)
#         
#         print(f"Remove line noise: {sid}")
#         raw.notch_filter(freqs=np.arange(50, 251, 50), notch_widths=5)
#         
#         if make_plot:
#             # Create second subplot for post-filtering PSD
#             raw.plot_psd(show=False, ax=ax2, fmin=l_freq, fmax=h_freq, average=False)
#             plt.title('After Filtering')
#             # Add overall title
#             plt.suptitle(f'Power Spectral Density - {sid}', fontsize=12)
#             # Adjust layout and display
#             plt.tight_layout()
#             current_fig = plt.gcf()  # Get current figure
#             plt.show()
#             
#             # Add to report if it exists
#             if hasattr(self, 'report') and self.report is not None:
#                 try:
#                     self.report.add_figure(
#                         current_fig, 
#                         title=f"PSD Before and After Filtering: {sid}", 
#                         section="Filtering", 
#                         tags=[sid, 'Filtering']
#                         )
#                     self.report.save(str(self.report_path), overwrite=True, open_browser=False)       
#                 except Exception as e:
#                     self.logger.warning(f"Failed to add PSD data to report for subject {sid}: {e}")
#             
#         #  Save as a .fif file
#         print(f"Save pre-processed rawfile: {cur_pptdata}")
#         self.save_fif(raw, 'filtered_raw')
# 
#         return raw
# =============================================================================
       
    def run_ICA(self, data, projectname, path_results, incl_report=False):
        """Run Independent Component Analysis (ICA) on MEG data.

        Parameters
        ----------
        data : mne.io.Raw
            Raw MEG data instance
        projectname : str
            Name of the project used for saving files
        path_results : str
            Path to save the ICA component figures and TSV file
        incl_report : bool, optional
            Whether to include ICA results in report, by default False
        
        Returns
        -------
        mne.io.Raw
            Raw data with ICA applied
        
        Notes
        -----
        - Resamples data to 200 Hz
        - Applies bandpass filter 1-95 Hz
        - Fits ICA with 20 components
        - Plots components and sources
        - Saves PNG of ICA components and sources
        - Exports excluded components to TSV file
        - Updates report if enabled
        """
        
        # NOTE: This function uses two different approaches for saving ICA results as images:
        # 1. For topographic maps (components), I can use ica.plot_components(show=False) which returns a
        #    standard matplotlib figure that can be directly saved.
        # 2. For time series data (sources), I create custom matplotlib figures rather than using the
        #    interactive MNE browser (ica.plot_sources()) because:
        #    - MNE's interactive Qt-based browser doesn't support direct saving to image files
        #    - Interactive browser plots can't be easily captured programmatically without user interaction 
        #    - Saving source plots is crucial since ICA results are non-deterministic and can vary between runs
        #    - Having a visual record of the actual ICA decomposition is important for reproducibility and
        #      helps maintain a record of what components were excluded and why
        #    This approach ensures we have permanent visual documentation of the ICA process for each subject.
        #
        # 3. A record of excluded components is saved in a TSV file for each subject:
        #    - Currently these are stored in a 'logging' folder inside the results directory
        #    - This organization may be revised in the future to integrate with a more comprehensive
        #      logging system 
        #    - In method sections it is usually reported some descriptive statistics of how many components per subject were rejected

        
        raw_resmpl = data.copy().pick('mag')
        raw_resmpl.resample(200) # dowsample to 200 Hz
        raw_resmpl.filter(1, 95) # band-pass filtert from 1 to 40 Hz
        
        ica = mne.preprocessing.ICA(method='fastica',
                                   random_state=97,
                                   n_components=20,
                                   verbose=True)
    
        ica.fit(raw_resmpl, verbose=True)
        ica.plot_components();
        
        # Save figure of ICA components
        subid = self.get_participant_id(data)
        fig = ica.plot_components(picks=range(ica.n_components_), show=False)
        fig_path = path_results +'/' +  projectname +'/' + f"{subid}_ICA_components.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved ICA components figure to: {fig_path}")
        ica.plot_sources(raw_resmpl, block=True)
        
        
        # Save sources using a custom matplotlib figure instead of the interactive browser
        sources = ica.get_sources(raw_resmpl)
        data_sources = sources.get_data()
    
        
        # Calculate the index corresponding to 10 seconds
        n_secs=60
        tsec_idx = int(n_secs * raw_resmpl.info['sfreq'])  # Convert 3 seconds to samples
        times = raw_resmpl.times[:tsec_idx]  # Only use times up to 3 seconds

    
        # Calculate how many components to put in each figure
        n_components = ica.n_components_
        half_components = n_components // 2
        first_half = range(0, half_components)
        second_half = range(half_components, n_components)

        # First figure - first half of components
        fig2a, axs_a = plt.subplots(nrows=len(first_half), figsize=(30, 15), sharex=True)
        for i, idx in enumerate(first_half):
            axs_a[i].plot(times, data_sources[idx, :tsec_idx])
            axs_a[i].set_title(f'ICA component {idx}')
            axs_a[i].set_xlim(0, n_secs)
            if i == len(first_half) - 1:
                axs_a[i].set_xlabel('Time (s)')

        fig2a.suptitle(f'ICA Sources (Components 0-{half_components-1}) - Subject {subid} (First 3 Seconds)')
        fig2a.tight_layout()
        fig2a_path = path_results +'/' + projectname +'/' + f"{subid}_ICA_sourcesA.png"
        fig2a.savefig(fig2a_path, dpi=300, bbox_inches='tight')
        plt.close(fig2a)
        print(f"Saved ICA sources figure (part 1) to: {fig2a_path}")
        
        # Second figure - second half of components
        fig2b, axs_b = plt.subplots(nrows=len(second_half), figsize=(30, 15), sharex=True)
        for i, idx in enumerate(second_half):
            axs_b[i].plot(times, data_sources[idx, :tsec_idx])
            axs_b[i].set_title(f'ICA component {idx}')
            axs_b[i].set_xlim(0, n_secs)
            if i == len(second_half) - 1:
                axs_b[i].set_xlabel('Time (s)')
        
        fig2b.suptitle(f'ICA Sources (Components {half_components}-{n_components-1}) - Subject {subid} (First 3 Seconds)')
        fig2b.tight_layout()
        fig2b_path = path_results +'/' + projectname +'/' + f"{subid}_ICA_sourcesB.png"
        fig2b.savefig(fig2b_path, dpi=300, bbox_inches='tight')
        plt.close(fig2b)
        print(f"Saved ICA sources figure (part 2) to: {fig2b_path}")
            
            
        # Create and save TSV file with excluded components information
        excluded_components = ica.exclude
        
        # Create a DataFrame with the component information
        comp_data = {
            'subject_id': [subid] * len(excluded_components) if excluded_components else [subid],
            'excluded_component': excluded_components if excluded_components else ['none']
        }
        
        # Create logging directory if it doesn't exist
        logging_dir = os.path.join(path_results, 'logging')
        os.makedirs(logging_dir, exist_ok=True)

        # Create DataFrame and save as TSV
        comp_df = pd.DataFrame(comp_data)
        tsv_path = os.path.join(logging_dir, f"{subid}_excluded_ICA_components.tsv")
        comp_df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Saved excluded components information to: {tsv_path}")

        
        ica.apply(data)
        
        
        # Add to report if it exists
        if hasattr(self, 'report') and self.report is not None and len(ica.exclude) > 0:
            try:
                self.report.add_ica(
                    ica=ica,
                    title="ICA Cleaning",
                    picks=ica.exclude,  # plot the excluded EOG components
                    inst=data,
                    n_jobs=None, 
                    tags=[subid, "ICA"]
                    )
            
                self.report.save(str(self.report_path), overwrite=True, open_browser=False)  
            except Exception as e:
                self.logger.warning(f"Failed to add ICA outputs to report for subject {subid}: {e}")
        
        #  Save as a .fif file
        print(f"Save post_ICA rawfile: {subid}")
        
        return data
    
    def run_annotate(self, raw):
        """Annotate raw data based on event triggers and breaks.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        make_plot : bool
            Whether to create event plots

        Returns
        -------
        mne.io.Raw
            Annotated raw data

        Notes
        -----
        - Creates annotations from events
        - Finds and annotates breaks between blocks
        - Drops bad channels
        - Updates report if enabled
        """
        
        # Retrieve subject id from raw filemane and get micropilot information from subj_info json file
        sid=raw.info['subject_info']['sid']
        
        #  Get micropilot information from subj_info json file
        mp_nr=next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs'] if subj['subj_id'] == sid), None)
        mapping=next((cur_mp['mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        mapping = {int(k): v for k, v in mapping.items()}
        known_bads=next((cur_mp['known_bad_chns'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        custom_mapping=next((cur_mp['custom_mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        
        # Annotate based on event triggers
        events = mne.find_events(raw)  # Annotate this line
        annot_from_events = mne.annotations_from_events(
            events=events,
            event_desc=mapping,
            sfreq=raw.info["sfreq"],
            # orig_time=raw.info["meas_date"], #TO DO: Double-check this
        )
        raw.set_annotations(annot_from_events)
        
        # Make plot of trigger events
        fig=mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=custom_mapping)
        plt.title(sid)
        plt.show()     
            
        # Add to report if it exists
        if hasattr(self, 'report') and self.report is not None:
            try:
                self.report.add_figure(fig=fig, title=f"Triggers overview: {sid}", section="Triggers overview", tags=[sid, "Triggers"])
                self.report.save(str(self.report_path), overwrite=True, open_browser=False)      
            except Exception as e:
                self.logger.warning(f"Failed to add raw data to report for subject {sid}: {e}")
            
        self.plot_infH_refH_loc(raw) 
        
        # Automatically find and annotate breaks between blocks
        break_annots = mne.preprocessing.annotate_break(
            raw=raw,
            min_break_duration=3,  # consider segments of at least 20 s duration
            t_start_after_previous=1,  # start annotation 5 s after end of previous one
            t_stop_before_next=1 # stop annotation 2 s before beginning of next one
            )
        raw.set_annotations(raw.annotations + break_annots)  # add to existing
        
        #Drop bad channels
        raw=self.drop_bad_ch(raw, known_bads)
        
        #  Save as a .fif file 
        self.save_fif(raw, descriptor='annotated_raw')
        
        return raw
    
    # def save_fif(self, raw, descriptor):
    #     fif_path=raw.info['subject_info']['fif_path']
    #     fif_name=raw.info['subject_info']['fif_name']
    #     new_filename = f"{fif_name}pipeline-{str(self.report_name)[:-5]}-{descriptor}.fif"
    #     new_filename =new_filename.replace('/', '').replace('\\', '')
    #     print(new_filename)
    #     print(fif_path)
    #     new_full_path = fif_path + new_filename
    #     raw.save(new_full_path, overwrite=True)
        
    def document_file_saving(self, filename, filepath, descriptor):
        """Document file saving details in the report.
    
        Parameters
        ----------
        filename : str
        Name of the saved file
        filepath : str
        Full path where the file was saved
        descriptor : str
        Description of the processing step
        
        Notes
        -----
        Adds to report:
            - Filename
            - File location
            - Saving timestamp
        """
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
        # Initialize list to collect summary lines
        summary_lines = []
        def add_line(line=""):
            print(line)
            summary_lines.append(line)
    
        add_line("\n=== FILE SAVING SUMMARY ===")
        add_line(f"\nProcessing Step: {descriptor}")
        add_line(f"Timestamp: {timestamp}")
        add_line(f"\nFile Details:")
        add_line(f"- Filename: {filename}")
        add_line(f"- Saved to: {filepath}")
        
        summary_lines = "\n".join(summary_lines)
        
        # Add to report if available
        if hasattr(self, 'report') and self.report is not None:
            try:
                self.report.add_html(
                    html=summary_lines,
                    title=f"File Saving: {descriptor}",
                    section="File Management",
                    tags=['file_saving', descriptor]
                    )
                self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                
            except Exception as e:
                    print(f"Warning: Failed to add file saving summary to report: {e}")
                    
        return summary_lines
                
    def save_fif(self, raw, descriptor, timestmp):
        fif_path = raw.info['subject_info']['fif_path']
        fif_name = raw.info['subject_info']['fif_name']
    
        # Get current date and time in a filename-friendly format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if timestmp == True:
            new_filename = f"{fif_name}pipeline-{str(self.report_name)}-{descriptor}-{timestamp}.fif"
        else:
            new_filename = f"{fif_name}file-{descriptor}.fif"
        
        new_filename = new_filename.replace('/', '').replace('\\', '')
        print(new_filename)
        print(fif_path)
    
        new_full_path = fif_path + new_filename
        raw.save(new_full_path, overwrite=True)
        
        # Print confirmation message with both filename and path
        print(f"\nFile '{new_filename}' successfully saved to:\n{new_full_path}")
        
        # Document in report
        self.document_file_saving(new_filename, new_full_path, descriptor)    
 
    def run_manual_inspect(self, data):
        """Manual inspection of Raw or Epochs/EpochsArray data with epoch rejection capability.
        
        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs or mne.EpochsArray
            Data to inspect

        Returns
        -------
        mne.io.Raw or mne.Epochs or mne.EpochsArray
            Inspected (and potentially cleaned) data object

        Notes
        -----
        Provides interactive plot for:
        - Raw data: Duration 10s, all channels
        - Epochs: Interactive epoch rejection
        Saves inspected data to file
        """
        # Check input type
        if isinstance(data, mne.io.Raw):
            # Handle Raw object
            raw_filename = data.filenames[0]
            match = re.search(r'(\d{8}_\d{6}_sub-P\d+)', raw_filename)
            cur_pptdata = str(match.group(1))
            
            # Plot Raw data
            data.plot(
                block=True,
                theme="light",
                show_scrollbars=True,
                duration=10,
                start=0,
                n_channels=len(data.ch_names),
                show_scalebars=True,
                scalings=dict(mag=1e-12, ref_meg=1e-12, stim=1),
                title=cur_pptdata,
                show_options=True,
                events=None
            )
            
            # Save Raw data
            path, filename = os.path.split(raw_filename)
            name, extension = os.path.splitext(filename)
            if name.endswith('_raw'):
                name = name[:-4]
            new_filename = f"{name}_inspected_raw{extension}"
            new_full_path = os.path.join(path, new_filename)
            data.save(new_full_path, overwrite=True)
            
        elif isinstance(data, (mne.Epochs, mne.EpochsArray)):
            # Store original number of epochs
            n_epochs_orig = len(data)
            
            # Get filename if available
            try:
                raw_filename = data.info['filename']
                match = re.search(r'(\d{8}_\d{6}_sub-P\d+)', raw_filename)
                cur_pptdata = str(match.group(1))
            except (KeyError, AttributeError):
                cur_pptdata = "EpochsArray"
            
            # Print instructions
            print("\nManual Inspection Instructions:")
            print("1. Mark bad epochs by clicking")
            print("2. Close the window when finished marking")
            print("3. Bad epochs will be automatically rejected")

            
            # Create a copy for modification
            data_clean = data.copy()
            
            # Plot Epochs data
            data_clean.plot(
                block=True,
                theme="light",
                show_scrollbars=True,
                n_channels=len(data_clean.ch_names),
                show_scalebars=True,
                scalings=dict(mag=1e-12, ref_meg=1e-12, stim=1),
                title=f"{cur_pptdata} - Mark bad epochs"
            )
            
            # Get and drop bad epochs
            bad_epochs = [idx for idx, log in enumerate(data_clean.drop_log) if 'USER' in log]
            
            # Print rejection summary
            if bad_epochs:
                n_dropped = len(bad_epochs)
                print(f"\nManual inspection complete:")
                print(f"- Originally: {n_epochs_orig} epochs")
                print(f"- Manually marked bad: {n_dropped} epochs ({n_dropped/n_epochs_orig*100:.1f}%)")
                print(f"- Remaining: {len(data_clean)} epochs")
                
                if hasattr(data_clean, 'event_id') and data_clean.event_id:
                    print("\nRejection by event type:")
                    events_orig = data.events[:, 2]
                    for event_name, event_id in data.event_id.items():
                        n_orig = np.sum(events_orig == event_id)
                        n_kept = np.sum(data_clean.events[:, 2] == event_id)
                        print(f"- {event_name}: {n_orig-n_kept} rejected out of {n_orig} "
                              f"({(n_orig-n_kept)/n_orig*100:.1f}%)")
            else:
                print("\nManual inspection complete: No epochs were marked as bad")
            
            # Save Epochs data if filename available
            if hasattr(data_clean, 'info') and 'filename' in data_clean.info:
                path, filename = os.path.split(raw_filename)
                name, extension = os.path.splitext(filename)
                if name.endswith('_epo'):
                    name = name[:-4]
                new_filename = f"{name}_inspected_epo{extension}"
                new_full_path = os.path.join(path, new_filename)
                data_clean.save(new_full_path, overwrite=True)
            
            return data_clean
            
        else:
            raise TypeError("Input must be an MNE Raw, Epochs, or EpochsArray object")
        
        return data
    
    def print_cleaning_summary(self, epochs_original, epochs_clean, cleaning_method='autoreject', 
                         ar_instance=None, threshold_params=None):
        """Print detailed cleaning summary for epoch rejection and add to report.
        Works with AutoReject, channel threshold, and manual cleaning methods.
        
        Parameters
        ----------
        epochs_original : mne.Epochs
            Original epochs before cleaning
        epochs_clean : mne.Epochs
            Cleaned epochs after rejection
        cleaning_method : str
            Method used for cleaning: 'autoreject', 'threshold', or 'manual'
        ar_instance : autoreject.AutoReject, optional
            The fitted AutoReject instance if using AutoReject
        threshold_params : dict, optional
            Dictionary containing threshold parameters if using threshold method:
            - 'q': quantile threshold
            - 'channel_fraction_threshold': fraction for bad channel marking
            
        Returns
        -------
        str
            Formatted summary text
        """
        # Initialize list to collect summary lines
        sid = epochs_original.info['subject_info']['sid']
        summary_lines = []
        def add_line(line=""):
            print(line)
            summary_lines.append(line)
        
        method_map = {
            'autoreject': 'AUTOREJECT',
            'threshold': 'THRESHOLD-BASED',
            'manual': 'MANUAL'
        }
        method_name = method_map.get(cleaning_method.lower(), cleaning_method.upper())
        add_line(f"\n=== {method_name} CLEANING SUMMARY ===")
        
        # Overall epoch rejection statistics
        n_epochs_total = len(epochs_original)
        n_epochs_kept = len(epochs_clean)
        n_epochs_rejected = n_epochs_total - n_epochs_kept
        
        add_line("\nOverall Epoch Statistics:")
        add_line(f"- Total epochs: {n_epochs_total}")
        add_line(f"- Rejected epochs: {n_epochs_rejected} ({n_epochs_rejected/n_epochs_total*100:.1f}%)")
        add_line(f"- Kept epochs: {n_epochs_kept} ({n_epochs_kept/n_epochs_total*100:.1f}%)")
        
        # Statistics by epoch type
        if hasattr(epochs_original, 'event_id') and epochs_original.event_id:
            add_line("\nStatistics by Epoch Type:")
            for event_name, event_id in epochs_original.event_id.items():
                # Original counts
                orig_events = epochs_original.events[:, 2] == event_id
                n_orig = np.sum(orig_events)
                
                # Kept counts (need to check if event still exists in cleaned epochs)
                if event_id in epochs_clean.event_id:
                    kept_events = epochs_clean.events[:, 2] == event_id
                    n_kept = np.sum(kept_events)
                else:
                    n_kept = 0
                
                n_rejected = n_orig - n_kept
                
                add_line(f"\n{event_name}:")
                add_line(f"- Total: {n_orig}")
                add_line(f"- Kept: {n_kept} ({n_kept/n_orig*100:.1f}%)")
                add_line(f"- Rejected: {n_rejected} ({n_rejected/n_orig*100:.1f}%)")
        
        # Channel Statistics
        orig_bads = epochs_original.info['bads']
        clean_bads = epochs_clean.info['bads']
        
        add_line("\nChannel Statistics:")
        add_line(f"- Total channels: {len(epochs_original.ch_names)}")
        add_line(f"- Originally marked bad: {len(orig_bads)}")
        add_line(f"- Additionally marked bad: {len(set(clean_bads) - set(orig_bads))}")
        add_line(f"- Total bad channels: {len(clean_bads)} "
                f"({len(clean_bads)/len(epochs_original.ch_names)*100:.1f}%)")
        
        if clean_bads:
            add_line("\nBad Channels:")
            for ch in clean_bads:
                if ch in orig_bads:
                    add_line(f"- {ch} (marked bad initially)")
                else:
                    marking_method = {
                        'autoreject': 'during autoreject',
                        'threshold': 'during threshold rejection',
                        'manual': 'during manual inspection'
                    }.get(cleaning_method.lower(), f'during {cleaning_method}')
                    add_line(f"- {ch} ({marking_method})")
        
        # Method-specific parameters
        if cleaning_method == 'autoreject' and ar_instance is not None:
            add_line("\nAutoReject Parameters:")
            add_line(f"- Consensus: {ar_instance.consensus}")
            add_line(f"- n_interpolate: {ar_instance.n_interpolate}")
            add_line(f"- thresh_method: {ar_instance.thresh_method}")
            
            # Get rejection thresholds if available
            if hasattr(ar_instance, 'thresholds_'):
                add_line("\nRejection Thresholds:")
                for ch_idx, thresh in enumerate(ar_instance.thresholds_):
                    ch_name = epochs_original.ch_names[ch_idx]
                    add_line(f"- {ch_name}: {thresh:.2f}")
                    
        elif cleaning_method == 'threshold' and threshold_params is not None:
            add_line("\nThreshold Parameters:")
            add_line(f"- Quantile threshold (q): {threshold_params['q']}")
            add_line(f"- Channel fraction threshold: {threshold_params['channel_fraction_threshold']}")
        
        # Drop log summary
        drop_log = epochs_clean.drop_log
        if drop_log:
            reasons = {}
            for log in drop_log:
                if log:  # if epoch was dropped
                    reason = log[0]  # get the first reason
                    reasons[reason] = reasons.get(reason, 0) + 1
            
            if reasons:
                add_line("\nDrop Reasons:")
                for reason, count in reasons.items():
                    add_line(f"- {reason}: {count} epochs ({count/n_epochs_total*100:.1f}%)")
        
        # Add to report if provided
        if hasattr(self, 'report') and self.report is not None:
            try:
                # Convert summary to HTML - using exact same conversion as working example
                html_content = []
                html_content.append("<div class='cleaning-summary'>")
                # Convert each line to HTML paragraph
                for line in summary_lines:
                    if line.startswith('==='): # Section headers
                        html_content.append(f"<h3>{line.strip('=').strip()}</h3>")
                    elif line.startswith('#'): # Sub-headers
                        html_content.append(f"<h4>{line.strip('#').strip()}</h4>")
                    elif not line.strip(): # Empty lines
                        html_content.append("<br>")
                    else: # Regular lines
                        # Handle indentation and special characters
                        line = line.replace(" ", "&nbsp;")
                        html_content.append(f"<p>{line}</p>")
                html_content.append("</div>")
                
                section_map = {
                    'autoreject': 'AutoReject',
                    'threshold': 'Threshold Rejection',
                    'manual': 'Manual Rejection'
                }
                section_name = section_map.get(cleaning_method.lower(), cleaning_method.title())
                
                self.report.add_html(
                    html="\n".join(html_content),  # Join the HTML content with newlines
                    title=f"{method_name} Epoch Rejection",
                    section=section_name,
                    tags=['cleaning_summary', cleaning_method.lower(), sid]
                )
                self.report.save(str(self.report_path), overwrite=True, open_browser=False)
    
            except Exception as e:
                print(f"Warning: Failed to add cleaning summary to report: {e}")
        
        return summary_lines
   
    def run_epoch(self, raw, tmin, tmax, baseline, reject, q_treshold=None, bad_chn_fraction=None):
        """Create epochs from raw data with specified parameters.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds
        reject : str
            Rejection method ('manual' or 'threshold')
        q_treshold : float
            If reject='treshold' then set the treshold for exclusion 
            (e.g. .97 will exclude the top .3 epochs with highest variance for that channel)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation
        bad_chn_fraction : float
            If reject='treshold' then set the fraction for channel rejection
            (e.g. .5 if for a given channel >.5 of epochs are reject, then the channel is rejected)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation

        Returns
        -------
        mne.Epochs
            Processed epochs

        Notes
        -----
        - Creates epochs from annotations
        - Applies baseline correction
        - Performs rejection based on specified method
        - Updates report with rejection statistics
        """
        
        sid = raw.info['subject_info']['sid']
        mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs'] if subj['subj_id'] == sid), None)
        cm = next((cur_mp['custom_mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        
        events_from_annot, event_dict = mne.events_from_annotations(raw, cm)
                
        epo = mne.Epochs(raw,
                      events_from_annot, 
                      event_dict,
                      tmin=tmin, 
                      tmax=tmax,
                      baseline=baseline,
                      proj=False,
                      picks='all',
                      detrend=1,
                      preload=True,
                      verbose=True)
        
        epo.info['subject_info'] = raw.info['subject_info']
        
        epochs_mag = epo.copy().pick_types(
            meg=True,          # Keep MEG channels
            ref_meg=False,     # Drop MEG reference channels
            stim=False,        # Drop stim channels
            exclude='bads'     # Drop bad channels
        )
        
        if reject == 'manual':
            # Store original epochs for comparison
            epochs_original = epochs_mag.copy()
            
            # Run manual inspection
            epochs_reject = self.run_manual_inspect(epochs_mag)
                        
            self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='manual')
            
            self.save_fif(epochs_reject, 'manualreject-all_epo')
            
            return epochs_reject
            
        elif reject == 'threshold':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_channel_thresholds(
                epochs_mag,
                q=q_treshold,                        # Set threshold at 97.5th percentile
                channel_fraction_threshold=bad_chn_fraction,  # Mark channels bad if >50% epochs are bad
                plot_results=True               # Show visualization
            )
            
            threshold_params = {'q': q_treshold,  # Quantile threshold for peak-to-peak amplitude
                                'channel_fraction_threshold': bad_chn_fraction  # Fraction of epochs that must be bad to mark a channel
                               }
            self.print_cleaning_summary(epochs_mag, epochs_reject, cleaning_method='threshold', threshold_params=threshold_params)
            
            self.save_fif(epochs_reject, 'tresholdreject-all_epo')
            
        elif reject == 'auto_reject':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_autoreject(self, epochs_mag, plot_results=True)
            
            # self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='autoreject', ar_instance=None)
            
            self.save_fif(epochs_reject, 'autoreject-all_epo')
        
        return epochs_reject
        """Create epochs from raw data with specified parameters.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds
        reject : str
            Rejection method ('manual' or 'threshold')
        q_treshold : float
            If reject='treshold' then set the treshold for exclusion 
            (e.g. .97 will exclude the top .3 epochs with highest variance for that channel)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation
        bad_chn_fraction : float
            If reject='treshold' then set the fraction for channel rejection
            (e.g. .5 if for a given channel >.5 of epochs are reject, then the channel is rejected)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation

        Returns
        -------
        mne.Epochs
            Processed epochs

        Notes
        -----
        - Creates epochs from annotations
        - Applies baseline correction
        - Performs rejection based on specified method
        - Updates report with rejection statistics
        """
        
        sid = raw.info['subject_info']['sid']
        mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs'] if subj['subj_id'] == sid), None)
        cm = next((cur_mp['custom_mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        
        events_from_annot, event_dict = mne.events_from_annotations(raw, cm)
                
        epo = mne.Epochs(raw,
                      events_from_annot, 
                      event_dict,
                      tmin=tmin, 
                      tmax=tmax,
                      baseline=baseline,
                      proj=False,
                      picks='all',
                      detrend=1,
                      preload=True,
                      verbose=True)
        
        epo.info['subject_info'] = raw.info['subject_info']
        
        epochs_mag = epo.copy().pick_types(
            meg=True,          # Keep MEG channels
            ref_meg=False,     # Drop MEG reference channels
            stim=False,        # Drop stim channels
            exclude='bads'     # Drop bad channels
        )
        
        if reject == 'manual':
            # Store original epochs for comparison
            epochs_original = epochs_mag.copy()
            
            # Run manual inspection
            epochs_reject = self.run_manual_inspect(epochs_mag)
                        
            self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='manual')
            
            self.save_fif(epochs_reject, 'manualreject-all_epo')
            
            return epochs_reject
            
        elif reject == 'threshold':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_channel_thresholds(
                epochs_mag,
                q=q_treshold,                        # Set threshold at 97.5th percentile
                channel_fraction_threshold=bad_chn_fraction,  # Mark channels bad if >50% epochs are bad
                plot_results=True               # Show visualization
            )
            
            threshold_params = {'q': q_treshold,  # Quantile threshold for peak-to-peak amplitude
                                'channel_fraction_threshold': bad_chn_fraction  # Fraction of epochs that must be bad to mark a channel
                               }
            self.print_cleaning_summary(epochs_mag, epochs_reject, cleaning_method='threshold', threshold_params=threshold_params)
            
            self.save_fif(epochs_reject, 'tresholdreject-all_epo')
            
        elif reject == 'auto_reject':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_autoreject(self, epochs_mag, plot_results=True)
            
            # self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='autoreject', ar_instance=None)
            
            self.save_fif(epochs_reject, 'autoreject-all_epo')
        
        return epochs_reject
        """Create epochs from raw data with specified parameters.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds
        reject : str
            Rejection method ('manual' or 'threshold')
        q_treshold : float
            If reject='treshold' then set the treshold for exclusion 
            (e.g. .97 will exclude the top .3 epochs with highest variance for that channel)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation
        bad_chn_fraction : float
            If reject='treshold' then set the fraction for channel rejection
            (e.g. .5 if for a given channel >.5 of epochs are reject, then the channel is rejected)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation

        Returns
        -------
        mne.Epochs
            Processed epochs

        Notes
        -----
        - Creates epochs from annotations
        - Applies baseline correction
        - Performs rejection based on specified method
        - Updates report with rejection statistics
        """
        
        sid = raw.info['subject_info']['sid']
        mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs'] if subj['subj_id'] == sid), None)
        cm = next((cur_mp['custom_mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        
        events_from_annot, event_dict = mne.events_from_annotations(raw, cm)
                
        epo = mne.Epochs(raw,
                      events_from_annot, 
                      event_dict,
                      tmin=tmin, 
                      tmax=tmax,
                      baseline=baseline,
                      proj=False,
                      picks='all',
                      detrend=1,
                      preload=True,
                      verbose=True)
        
        epo.info['subject_info'] = raw.info['subject_info']
        
        epochs_mag = epo.copy().pick_types(
            meg=True,          # Keep MEG channels
            ref_meg=False,     # Drop MEG reference channels
            stim=False,        # Drop stim channels
            exclude='bads'     # Drop bad channels
        )
        
        if reject == 'manual':
            # Store original epochs for comparison
            epochs_original = epochs_mag.copy()
            
            # Run manual inspection
            epochs_reject = self.run_manual_inspect(epochs_mag)
                        
            self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='manual')
            
            self.save_fif(epochs_reject, 'manualreject-all_epo')
            
            return epochs_reject
            
        elif reject == 'threshold':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_channel_thresholds(
                epochs_mag,
                q=q_treshold,                        # Set threshold at 97.5th percentile
                channel_fraction_threshold=bad_chn_fraction,  # Mark channels bad if >50% epochs are bad
                plot_results=True               # Show visualization
            )
            
            threshold_params = {'q': q_treshold,  # Quantile threshold for peak-to-peak amplitude
                                'channel_fraction_threshold': bad_chn_fraction  # Fraction of epochs that must be bad to mark a channel
                               }
            self.print_cleaning_summary(epochs_mag, epochs_reject, cleaning_method='threshold', threshold_params=threshold_params)
            
            self.save_fif(epochs_reject, 'tresholdreject-all_epo')
            
        elif reject == 'auto_reject':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_autoreject(self, epochs_mag, plot_results=True)
            
            # self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='autoreject', ar_instance=None)
            
            self.save_fif(epochs_reject, 'autoreject-all_epo')
        
        return epochs_reject
        """Create epochs from raw data with specified parameters.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds
        reject : str
            Rejection method ('manual' or 'threshold')
        q_treshold : float
            If reject='treshold' then set the treshold for exclusion 
            (e.g. .97 will exclude the top .3 epochs with highest variance for that channel)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation
        bad_chn_fraction : float
            If reject='treshold' then set the fraction for channel rejection
            (e.g. .5 if for a given channel >.5 of epochs are reject, then the channel is rejected)
            This is only temporary, while we resolve the issue with using auto-reject without interpolation

        Returns
        -------
        mne.Epochs
            Processed epochs

        Notes
        -----
        - Creates epochs from annotations
        - Applies baseline correction
        - Performs rejection based on specified method
        - Updates report with rejection statistics
        """
        
        sid = raw.info['subject_info']['sid']
        mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs'] if subj['subj_id'] == sid), None)
        cm = next((cur_mp['custom_mapping'] for cur_mp in self.subj_info['micropilots'] if cur_mp['micro_pilot_nr'] == mp_nr), None)
        
        events_from_annot, event_dict = mne.events_from_annotations(raw, cm)
                
        epo = mne.Epochs(raw,
                      events_from_annot, 
                      event_dict,
                      tmin=tmin, 
                      tmax=tmax,
                      baseline=baseline,
                      proj=False,
                      picks='all',
                      detrend=1,
                      preload=True,
                      verbose=True)
        
        epo.info['subject_info'] = raw.info['subject_info']
        
        epochs_mag = epo.copy().pick_types(
            meg=True,          # Keep MEG channels
            ref_meg=False,     # Drop MEG reference channels
            stim=False,        # Drop stim channels
            exclude='bads'     # Drop bad channels
        )
        
        if reject == 'manual':
            # Store original epochs for comparison
            epochs_original = epochs_mag.copy()
            
            # Run manual inspection
            epochs_reject = self.run_manual_inspect(epochs_mag)
                        
            self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='manual')
            
            self.save_fif(epochs_reject, 'manualreject-all_epo')
            
            return epochs_reject
            
        elif reject == 'threshold':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_channel_thresholds(
                epochs_mag,
                q=q_treshold,                        # Set threshold at 97.5th percentile
                channel_fraction_threshold=bad_chn_fraction,  # Mark channels bad if >50% epochs are bad
                plot_results=True               # Show visualization
            )
            
            threshold_params = {'q': q_treshold,  # Quantile threshold for peak-to-peak amplitude
                                'channel_fraction_threshold': bad_chn_fraction  # Fraction of epochs that must be bad to mark a channel
                               }
            self.print_cleaning_summary(epochs_mag, epochs_reject, cleaning_method='threshold', threshold_params=threshold_params)
            
            self.save_fif(epochs_reject, 'tresholdreject-all_epo')
            
        elif reject == 'auto_reject':
            
            epochs_reject, bad_epochs, bad_channels = self.clean_epochs_with_autoreject(self, epochs_mag, plot_results=True)
            
            # self.print_cleaning_summary(epochs_original, epochs_reject, cleaning_method='autoreject', ar_instance=None)
            
            self.save_fif(epochs_reject, 'autoreject-all_epo')
        
        return epochs_reject
         
    
        def process_events(self, events_from_annot, event_dict, new_events):
            """
            Process event annotations to classify them into new event categories.
            
            Parameters:
            -----------
            events_from_annot : numpy.ndarray
                Array of shape (n_events, 3) containing [timestamp, 0, event_code]
            event_dict : dict
                Dictionary mapping event names to their numerical codes
            new_events : list
                List of new event categories to classify events into
                
            Returns:
            --------
            tuple
                (processed_events, new_event_dict) where:
                - processed_events: numpy.ndarray with modified event codes
                - new_event_dict: dictionary mapping new categories to their codes
            """
            # Create reverse mapping from codes to event names
            code_to_event = {code: event for event, code in event_dict.items()}
            
            # Create a copy of the events array to modify
            processed_events = events_from_annot.copy()
            
            # Create new event dictionary
            new_event_dict = {event: idx + 1 for idx, event in enumerate(new_events)}
            
            # Process each event
            for i in range(len(processed_events)):
                event_code = processed_events[i, 2]
                if event_code in code_to_event:
                    event_name = code_to_event[event_code]
                    # Check if event contains any of the new event categories
                    for new_event in new_events:
                        if new_event in event_name:
                            processed_events[i, 2] = new_event_dict[new_event]
                            break
            
            return processed_events, new_event_dict

    
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

    def create_montage(self, raw):
        """
        Create a DigMontage object from MEG sensor locations in raw data.
    
        Extracts MEG sensor positions from raw.info and creates a corresponding
        digitization montage in head coordinate frame.
    
        Parameters
        ----------
        raw : mne.io.Raw
            Raw instance containing sensor location information in info['chs']
    
        Returns
        -------
        montage : mne.channels.DigMontage
            Digitization montage containing sensor locations
    
        Notes
        -----
        - Only includes channels with non-zero positions
        - Assumes coordinates are in head frame
        - Only processes MEG channels (excludes other sensor types)
        """
        indices =  mne.pick_types(raw.info, meg=False, ref_meg=True, stim=True)  # example indices
        raw_montage=raw.drop_channels([raw.ch_names[i] for i in indices])
        raw_montage=self.align_topo_locs(raw_montage)
        
        ch_pos = {}
        for idx, ch_name in enumerate(raw_montage.ch_names):
            if hasattr(raw_montage.info['chs'][idx]['loc'], 'any'):  # Check if position exists
                pos = raw_montage.info['chs'][idx]['loc'][:3]  # First 3 values are x,y,z coordinates
                ch_pos[ch_name] = pos
        
        # Create montage from positions
        from mne.channels import make_dig_montage
        montage = make_dig_montage(ch_pos, coord_frame='unknown')
        
        # Verify the montage
        print("\nMontage details:")
        print(montage)
        
        # Plot to visualize sensor positions
        montage.plot()
        
        return montage

    def create_sphere_from_sensors(self, raw):
        """Create sphere model based on sensor positions and visualize it"""        
        # Get MEG channel positions
        meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False, stim=False)
        pos = np.array([raw.info['chs'][idx]['loc'][:3] for idx in meg_picks])
        
        # Calculate center and radius
        center = np.mean(pos, axis=0)
        radius = np.median(np.linalg.norm(pos - center, axis=1))
        
        # Create sphere model
        sphere = make_sphere_model(
            r0=center,  # Center of the sphere
            head_radius=radius  # Radius of the sphere
        )
        
        # Create the visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sensor positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', marker='o', label='Sensors')
        
        # # Create sphere surface
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        # y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        # z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        # # Plot sphere surface with transparency
        # ax.plot_surface(x, y, z, color='r', alpha=0.1)
        
        # Plot sphere center
        ax.scatter([center[0]], [center[1]], [center[2]], 
                   c='r', marker='*', s=200, label='Sphere Center')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Sensor Positions and Fitted Sphere')
        
        # Add legend
        ax.legend()
        
        # Make the plot more visually appealing
        ax.view_init(elev=20, azim=45)
        
        # Ensure equal aspect ratio
        max_range = np.array([pos[:, 0].max()-pos[:, 0].min(),
                             pos[:, 1].max()-pos[:, 1].min(),
                             pos[:, 2].max()-pos[:, 2].min()]).max() / 2.0
        mid_x = (pos[:, 0].max()+pos[:, 0].min()) * 0.5
        mid_y = (pos[:, 1].max()+pos[:, 1].min()) * 0.5
        mid_z = (pos[:, 2].max()+pos[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
        
        print(f"Sphere center: {sphere['r0']}")
        print(f"Sphere radius: {radius:.3f} m")
        
        return sphere    
    
    def run_interpolate(self, raw):
        
        raw.plot(title='pre_interpolate')
        raw_montage=raw.copy()
        montage=self.create_montage(raw_montage)
        del raw_montage
        
        # Apply to your raw data
        raw=raw.set_montage(montage)
        raw.plot(title='pos_montage')

        
        # Create sphere and use its origin
        sphere = self.create_sphere_from_sensors(raw)
        
        raw=raw.interpolate_bads(
             reset_bads=False,
             mode='accurate',
             origin=sphere['r0'])

        # raw_interpolated.plot()
        # Clear the bads list
        raw.info['bads'] = []
        
        return raw
            
    
    def collapse_epochs(self, epochs, new_label):
        """Collapse multiple epoch categories into a single category.
        
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object with multiple categories
        new_label : str, optional
            The new label for all epochs, by default 'syl'
            
        Returns
        -------
        mne.Epochs
            New epochs object with single category
        """
        # Create a new event_id dictionary with single category
        new_event_id = {new_label: 1}
        
        # Create new events array with all events marked as 1
        new_events = epochs.events.copy()
        new_events[:, 2] = 1
        
        # Create new epochs object with collapsed events
        epochs_collapsed = mne.EpochsArray(
            data=epochs.get_data(),
            info=epochs.info.copy(),
            events=new_events,
            event_id=new_event_id,
            tmin=epochs.tmin,
            baseline=epochs.baseline
        )
        
        return epochs_collapsed
    
    def concatenate_epochs_properly(self, epochs_list):
        """Concatenate epochs while maintaining correct event counts.
        
        Parameters
        ----------
        epochs_list : list
            List of epochs/epochsarray objects to concatenate
        
        Returns
        -------
        mne.EpochsArray
            Concatenated epochs with correct event counts

        Raises
        ------
        ValueError
            If epochs have different time windows
        """
        # First check if all epochs have the same properties
        ref_epochs = epochs_list[0]
        for epochs in epochs_list[1:]:
            if epochs.tmin != ref_epochs.tmin or epochs.tmax != ref_epochs.tmax:
                raise ValueError("All epochs must have the same time window")
            if epochs.info['sfreq'] != ref_epochs.info['sfreq']:
                raise ValueError("All epochs must have the same sampling frequency")
    
        # Combine data
        all_data = []
        all_events = []
        current_event_id = {}
        event_id_counter = 1
    
        # Process each epochs object
        for epochs in epochs_list:
            data = epochs.get_data()
            all_data.append(data)
            
            # Get events and adjust their IDs
            events = epochs.events.copy()
            
            # Map the event IDs
            for event_type, old_id in epochs.event_id.items():
                if event_type not in current_event_id:
                    current_event_id[event_type] = event_id_counter
                    event_id_counter += 1
                
                # Update the event codes
                events[epochs.events[:, 2] == old_id, 2] = current_event_id[event_type]
            
            all_events.append(events)
    
        # Concatenate all data
        data = np.concatenate(all_data, axis=0)
        
        # Concatenate and adjust events
        events = np.concatenate(all_events, axis=0)
        
        # Create the concatenated epochs
        epochs_concat = mne.EpochsArray(
            data=data,
            info=ref_epochs.info.copy(),
            events=events,
            event_id=current_event_id,
            tmin=ref_epochs.tmin,
            baseline=ref_epochs.baseline
        )
        
        # Print concatenation summary
        print("\nConcatenation Summary:")
        print("Original epochs:")
        for i, epochs in enumerate(epochs_list):
            print(f"Epochs {i+1}: {len(epochs)} events")
            for event_type, count in epochs.event_id.items():
                print(f"  {event_type}: {len(epochs[event_type])} events")
        
        print("\nConcatenated epochs:")
        print(f"Total: {len(epochs_concat)} events")
        for event_type in epochs_concat.event_id.keys():
            print(f"  {event_type}: {len(epochs_concat[event_type])} events")
        
        return epochs_concat

    
    def run_evoked(self, epo, lst_cond, equalize, highpass, lowpass):
        """
        Process and visualize evoked responses from epochs data.
    
        Generates butterfly plots, topomaps, and joint plots for each condition and their pairwise differences.
        Results are saved to a report if one exists.
    
        Parameters
        ----------
        epo : mne.Epochs
            The epochs object containing MEG data
        lst_cond : list of str
            List of condition names to process
        equalize : bool
            Whether to equalize epoch counts across conditions using random selection
        highpass : None, int
            highpass filter applied to all epochs
            Note: This should not be lower than the lowpass filter previously used at preprocessing.run_filtering
            TODO: Manage filter setting on main code to prevent mismatches
        lowpass : None, int
            highpass filter applied to all epochs
            Note: This should not be higher than the lowpass filter previously used at preprocessing.run_filtering
            TODO: Manage filter setting on main code to prevent mismatches
    
        Returns
        -------
        evo_data : mne.Evoked 
            Combined evoked data across all conditions
    
        Notes
        -----
        Generates multiple visualization plots:
        - Butterfly plots for each condition
        - Topomaps at predefined timepoints
        - Joint plots combining timecourse and topography
        - Pairwise difference plots between conditions
    
        All plots are added to the report if report_path is set.
        """    
        lst_evo = []
        lst_eppo=[]
        sid=epo.info['subject_info']['sid']
        
        for cnd in lst_cond:
            epochs_cur=epo[cnd]
            epochs_epo = self.collapse_epochs(epochs_cur, new_label=cnd)
            lst_eppo.append(epochs_epo)

        if equalize==True:
            mne.epochs.equalize_epoch_counts(lst_eppo, method='random', random_state=42)
        
        lst_cnt= self.concatenate_epochs_properly(lst_eppo)
        lst_evo = lst_cnt.average(method='mean', by_event_type=True)
           
        evo_data = mne.combine_evoked(lst_evo, weights='equal').filter(highpass, lowpass)
        evo_dict = dict(zip(lst_cond, lst_evo))
        
        # Generate all possible pairs of conditions if more than one condition
        pairs = []
        if len(lst_cond) > 1:
            pairs = [(lst_cond[i], lst_cond[j]) 
                    for i in range(len(lst_cond)) 
                    for j in range(i+1, len(lst_cond))]
            
        # Get evenly spaced times across the entire epoch:
        start_time = epo.times[0]  # First timepoint
        end_time = epo.times[-1] 
        interval = 0.1  # 100ms intervals
        times = np.arange(start_time, end_time, interval)
        
        # # Butterfly plots
        # # TODO: Times for joint plot should not be hardcoded
        #times = [0,.1,.2,.3,.4,.5,.6]
        # for i in range(len(evo_dict)):
        #     current_condition = list(evo_dict.keys())[i]
        #     fig1=evo_dict[current_condition].plot(gfp=True, titles=current_condition, exclude='bads')
        #     # Add to report if it exists
        #     if hasattr(self, 'report') and self.report is not None:
        #         try:
        #             self.report.add_figure(fig=fig1, title=f"Butterfly {current_condition}", 
        #                          section='Evoked', tags=[sid, "Butterfly_ERF"])
        #             self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #         except Exception as e:
        #             self.logger.warning(f"Failed to Butterfly plot: {e}")

        # # Add pairwise difference butterfly plots
        # for cond1, cond2 in pairs:
        #     diff_evoked = evo_dict[cond1].copy()
        #     diff_evoked.data = evo_dict[cond1].data - evo_dict[cond2].data
        #     fig1_diff = diff_evoked.plot(gfp=True, titles=f'{cond1} - {cond2}', exclude='bads')
        #     if hasattr(self, 'report') and self.report is not None:
        #         try:
        #             self.report.add_figure(fig=fig1_diff, title=f"Butterfly Difference {cond1}-{cond2}", 
        #                          section='Evoked', tags=[sid, "Butterfly_ERF_Difference"])
        #             self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #         except Exception as e:
        #             self.logger.warning(f"Failed to Butterfly plot: {e}")
            
        # # Topomaps
        # for i in range(len(evo_dict)):
        #     current_condition = list(evo_dict.keys())[i]
        #     fig2 = evo_dict[current_condition].plot_topomap(
        #         times, 
        #         ch_type="mag"
        #     )
        #     fig2.suptitle(f'{current_condition}', fontsize=16)
        #     if hasattr(self, 'report') and self.report is not None:
        #         try:
        #             self.report.add_figure(fig=fig2, title=f"Topomap {current_condition}", 
        #                          section='Evoked', tags=[sid, "Topomap"])
        #             self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #         except Exception as e:
        #                 self.logger.warning(f"Failed to Topomap plot: {e}")
        
        # # Add pairwise difference topomaps
        # for cond1, cond2 in pairs:
        #     diff_evoked = evo_dict[cond1].copy()
        #     diff_evoked.data = evo_dict[cond1].data - evo_dict[cond2].data
        #     fig2_diff = diff_evoked.plot_topomap(
        #         times, 
        #         ch_type="mag"
        #     )
        #     fig2_diff.suptitle(f'{cond1} - {cond2}', fontsize=16)
        #     if hasattr(self, 'report') and self.report is not None:
        #         try:
        #             self.report.add_figure(fig=fig2_diff, title=f"Topomap Difference {cond1}-{cond2}", 
        #                          section='Evoked', tags=[sid, "Topomap_Difference"])
        #             self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #         except Exception as e:
        #                 self.logger.warning(f"Failed to Topomap plot: {e}")
            
        # Joint plots
        # TODO: Plotting limits should not be hardcoded
        topomap_args = {
            'vlim': (-300, 300),   
            'cmap': 'RdBu_r'     
        }
        ts_args = dict(ylim=dict(mag=[-650, 650]))
        
        for i in range(len(evo_dict)):
            current_condition = list(evo_dict.keys())[i]
            fig3=evo_dict[current_condition].plot_joint(times=times, exclude='bads', 
                                                      title=current_condition, picks="mag", 
                                                      ts_args=ts_args, topomap_args=topomap_args)
            
            if hasattr(self, 'report') and self.report is not None:
                try:
                    self.report.add_figure(fig=fig3, title=f"Joint plot {current_condition}", 
                                 section='Evoked', tags=[sid, "Joint_ERF"])
                    self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                except Exception as e:
                        self.logger.warning(f"Failed to Joint plot: {e}")
        
        # # Add pairwise difference joint plots
        # for cond1, cond2 in pairs:
        #     diff_evoked = evo_dict[cond1].copy()
        #     diff_evoked.data = evo_dict[cond1].data - evo_dict[cond2].data
        #     fig3_diff = diff_evoked.plot_joint(times=times, exclude='bads', 
        #                                      title=f'{cond1} - {cond2}', picks="mag", 
        #                                      ts_args=ts_args, topomap_args=topomap_args)
        #     if hasattr(self, 'report') and self.report is not None:
        #         try:
        #             self.report.add_figure(fig=fig3_diff, title=f"Joint plot Difference {cond1}-{cond2}", 
        #                          section='Evoked', tags=[sid, "Joint_ERF_Difference"])
        #             self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #         except Exception as e:
        #                 self.logger.warning(f"Failed to Joint plot: {e}")
        
        # # Compare evokeds plot
        # fig4=mne.viz.plot_compare_evokeds(
        #     evo_dict,
        #     picks="mag",
        #     axes="topo",
        #     )
        
        # if hasattr(self, 'report') and self.report is not None:
        #     try:
        #         self.report.add_figure(fig=fig4, title=f"Topomap ERF", section='Evoked', 
        #                      tags=[sid, 'Topomap_Chan_ERF'])
        #         self.report.save(str(self.report_path), overwrite=True, open_browser=False)
        #     except Exception as e:
        #             self.logger.warning(f"Failed to Topomap plot: {e}")
                    
        # Save as a .fif file
        print(f"Save pre-processed rawfile: {sid}")
        
        if equalize==True:
            self.save_fif(evo_data, 'eq_evo')
        else:
            self.save_fif(evo_data, 'all_evo')
            
        return evo_data

    
    def check_meg_channel_type(self, channel_string):
        """
        Determine the type of MEG channel from its string description.
    
        Parameters
        ----------
        channel_string : str
            String containing channel type information
    
        Returns
        -------
        str
            One of:
            - 'REF_MEG_CH': Reference MEG channel
            - 'MEG_CH': Regular MEG channel
            - 'Neither': Neither type
    
        Notes
        -----
        Used for categorizing channels in visualization and processing functions
        to apply appropriate handling based on channel type.
        """
        if 'REF_MEG_CH' in channel_string:
            return 'REF_MEG_CH'
        elif 'MEG_CH' in channel_string:
            return 'MEG_CH'
        else:
            return 'Neither'
          
    def create_montage_from_raw(self, raw):
        """
        Create a DigMontage object from MEG sensor locations in raw data.
    
        Extracts MEG sensor positions from raw.info and creates a corresponding
        digitization montage in head coordinate frame.
    
        Parameters
        ----------
        raw : mne.io.Raw
            Raw instance containing sensor location information in info['chs']
    
        Returns
        -------
        montage : mne.channels.DigMontage
            Digitization montage containing sensor locations
    
        Notes
        -----
        - Only includes channels with non-zero positions
        - Assumes coordinates are in head frame
        - Only processes MEG channels (excludes other sensor types)
        """
        # Get channel positions and names
        ch_pos = {}
        for idx, ch in enumerate(raw.info['chs']):
            if str(ch['kind']).endswith('(FIFFV_MEG_CH)'):
                # Extract the sensor location
                loc = ch['loc'][:3]  # Take first 3 values (x, y, z coordinates)
                if not np.all(loc == 0):  # Only include non-zero positions
                    ch_pos[ch['ch_name']] = loc
        
        # Create the DigMontage
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame='head'  # Assuming the coordinates are in head frame
        )
        
        return montage
    
    def run_refreg(self, raw, mode):
        
        print(f"Apply reference regression:")
        # compute the PSD for later using 1 Hz resolution
        psd_kwargs = dict(fmin=.1, fmax=95, n_fft=int(round(raw.info["sfreq"])))
        psd_pre = raw.compute_psd(**psd_kwargs)
       
        if mode=='ref_ch':
            # Get the indices of ref_meg channels
            # filter and regress 
            # raw.filter(None, 3, picks="ref_meg")
            raw_reg=raw.copy()
            # Get the indices of ref_meg channels
            ref_meg_idx = mne.pick_types(raw_reg.info, ref_meg=True)
            ref_meg_channels = [raw_reg.ch_names[idx] for idx in ref_meg_idx]
           
            regress = mne.preprocessing.EOGRegression(picks='meg', picks_artifact="ref_meg")
            regress.fit(raw_reg)
            regraw = regress.apply(raw_reg, copy=True)
           
            # compute the psd of the regressed data
            psd_post_reg = regraw.compute_psd(**psd_kwargs)
           
            shielding = 10 * np.log10(psd_pre[:] / psd_post_reg[:])
            sid=regraw.info['subject_info']['sid']
            plot_kwargs = dict(lw=1, alpha=0.5)
            
            # Create figure with 3 subplots in one row
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot PSD pre
            _raw=raw.copy()
            _raw=self.align_topo_locs(_raw)
            _raw.plot_psd(fmin=.1, fmax=95, ax=ax1)
            ax1.set_title('Pre- reference regression', fontsize=16)
            
            # Plot PSD post
            _regraw=regraw.copy()
            _regraw=self.align_topo_locs(_regraw)
            _regraw.plot_psd(fmin=.1, fmax=95, ax=ax2)
            ax2.set_title('Post- reference regression', fontsize=16)
            
            # Plot shielding factor
            ax3.plot(psd_post_reg.freqs, shielding.T, **plot_kwargs)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Add horizontal line at y=0
            ax3.grid(True, ls=":")
            ax3.set(
                xlim=(0, 40),
                title=f"Reference Regression Shielding: {sid}",
                xlabel="Frequency (Hz)",
                ylabel="Shielding (dB)"
            )
            
            plt.tight_layout()
               
            # Add to report if it exists
            if hasattr(self, 'report') and self.report is not None:
                try:
                    self.report.add_figure(fig=fig, title=f"Reference Regression: {sid}", section="Reference Regression", tags=[sid, "Reference_Reg"])
                    self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                except Exception as e:
                    self.logger.warning(f"Failed to add channels Reference Regression plot to report for subject {sid}: {e}")
    
        
            return regraw

        if mode=='ref_ICAsep':
            try:
                raw_tog = raw.copy()
                ica_kwargs = dict(
                    method="picard",
                    fit_params=dict(tol=1e-4),  # use a high tol here for speed
                    )
                all_picks = mne.pick_types(raw_tog.info, meg=True, ref_meg=True)
                ica_tog = mne.preprocessing.ICA(n_components=20, max_iter="auto", allow_ref_meg=True, **ica_kwargs)
                ica_tog.fit(raw_tog, picks=all_picks)
    
                raw_sep = raw.copy()
    
                # Do ICA only on the reference channels.
                ref_picks = mne.pick_types(raw_sep.info, meg=False, ref_meg=True)
                ica_ref = mne.preprocessing.ICA(n_components=2, max_iter="auto", allow_ref_meg=True, **ica_kwargs)
                ica_ref.fit(raw_sep, picks=ref_picks)
                
                # Do ICA on both reference and standard channels. Here, we can just reuse
                # ica_tog from the section above.
                ica_sep = ica_tog.copy()
                
                # Extract the time courses of these components and add them as channels
                # to the raw data. Think of them the same way as EOG/EKG channels, but instead
                # of giving info about eye movements/cardiac activity, they give info about
                # external magnetic noise.
                ref_comps = ica_ref.get_sources(raw_sep)
                for c in ref_comps.ch_names:  # they need to have REF_ prefix to be recognised
                    ref_comps.rename_channels({c: "REF_" + c})
                raw_sep.add_channels([ref_comps])
                
                # Now that we have our noise channels, we run the separate algorithm.
                bad_comps, scores = ica_sep.find_bads_ref(raw_sep, method="separate")
                
                # Plot scores with bad components marked.
                ica_sep.plot_scores(scores, bad_comps)
                
                # Examine the properties of removed components.
                ica_sep.plot_properties(raw_sep, picks=bad_comps)
                
                # Remove the components.
                raw_sep = ica_sep.apply(raw_sep, exclude=bad_comps)
    
                # compute the psd of the regressed data
                psd_post_reg = raw_sep.compute_psd(**psd_kwargs)
               
                #make_plot
                shielding = 10 * np.log10(psd_pre[:] / psd_post_reg[:])
                sid=raw_sep.info['subject_info']['sid']
                plot_kwargs = dict(lw=1, alpha=0.5)
                
                # Create figure with 3 subplots in one row
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot PSD pre
                _raw=raw.copy()
                _raw.plot_psd(fmin=.1, fmax=100, ax=ax1)
                ax1.set_title('Pre- ICA reference regression', fontsize=16)
                
                # Plot PSD post
                _raw_sep=raw_sep.copy()
                _raw_sep.plot_psd(fmin=.1, fmax=100, ax=ax2)
                ax2.set_title('Post- ICA reference regression', fontsize=16)
                
                # Plot shielding factor
                ax3.plot(psd_post_reg.freqs, shielding.T, **plot_kwargs)
                ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Add horizontal line at y=0
                ax3.grid(True, ls=":")
                ax3.set(
                    xlim=(0, 40),
                    title=f"ICA Reference Regression Shielding: {sid}",
                    xlabel="Frequency (Hz)",
                    ylabel="Shielding (dB)"
                )
                
                plt.tight_layout()
                   
                # Add to report if it exists
                if hasattr(self, 'report') and self.report is not None:
                    try:
                        self.report.add_figure(fig=fig, title=f"ICA Reference Regression: {sid}", section="ICA Reference Regression", tags=[sid, "ICA_Reference_Reg"])
                        self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                    except Exception as e:
                        self.logger.warning(f"Failed to add ICA Reference Regression plot to report for subject {sid}: {e}")
        
            
                return raw_sep
            except Exception as e:
              self.logger.warning(f"Failed to add ICA Reference Regression plot to report for subject {sid}: {e}")
              
    def compute_channel_specific_thresholds(self, epochs, q):
        """
        Compute amplitude thresholds for each channel based on their data distribution.
    
        Calculates peak-to-peak values for each channel and epoch, then determines
        channel-specific thresholds based on the specified quantile.
    
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object to analyze
        q : float
            Quantile value (0-1) to use for threshold determination
    
        Returns
        -------
        thresholds : ndarray, shape (n_channels,)
            Channel-specific threshold values
        peak_to_peak : ndarray, shape (n_epochs, n_channels)
            Peak-to-peak values for each epoch and channel
        bad_epochs_per_channel : ndarray, shape (n_channels, n_epochs)
            Boolean array marking epochs exceeding threshold for each channel
    
        Notes
        -----
        Prints summary statistics for each channel, including the number and
        percentage of epochs exceeding the threshold.
        """
        # Calculate peak-to-peak values
        data = epochs.get_data()
        peak_to_peak = np.ptp(data, axis=2)  # shape: (n_epochs, n_channels)
        
        # Calculate channel-specific thresholds
        thresholds = np.zeros(len(epochs.ch_names))
        bad_epochs_per_channel = np.zeros((len(epochs.ch_names), len(epochs)), dtype=bool)
        
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            # Get this channel's peak-to-peak values
            ch_data = peak_to_peak[:, ch_idx]
            
            # Compute threshold for this channel
            threshold = np.quantile(ch_data, q)
            thresholds[ch_idx] = threshold
            
            # Mark bad epochs for this channel
            bad_epochs = ch_data > threshold
            bad_epochs_per_channel[ch_idx] = bad_epochs
            
            # Print summary for this channel
            n_bad = np.sum(bad_epochs)
            print(f"Channel {ch_name}: {n_bad}/{len(epochs)} epochs exceed threshold "
                  f"({n_bad/len(epochs)*100:.1f}%)")
        
        return thresholds, peak_to_peak, bad_epochs_per_channel

    def plot_channel_thresholds(self, epochs, thresholds, peak_to_peak, bad_epochs_per_channel):
        """
        Visualize channel-specific thresholds and their effects on epoch rejection.
    
        Creates a two-panel figure showing:
        1. Channel-wise amplitude distributions with threshold lines
        2. Heatmap of bad epochs per channel
    
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object being analyzed
        thresholds : ndarray, shape (n_channels,)
            Channel-specific threshold values
        peak_to_peak : ndarray, shape (n_epochs, n_channels)
            Peak-to-peak values for each epoch and channel
        bad_epochs_per_channel : ndarray, shape (n_channels, n_epochs)
            Boolean array marking epochs exceeding threshold for each channel
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the visualization plots
    
        Notes
        -----
        The upper panel shows violin plots of peak-to-peak distributions with:
        - Red horizontal lines indicating thresholds
        - Red dots marking epochs exceeding thresholds
        
        The lower panel shows a heatmap where:
        - Rows represent channels
        - Columns represent epochs
        - Red cells indicate bad epochs
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Plot 1: Channel-wise distributions with thresholds
        ax1 = fig.add_subplot(gs[0])
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            violin = ax1.violinplot(peak_to_peak[:, ch_idx], positions=[ch_idx])
            
            # Plot threshold
            ax1.hlines(thresholds[ch_idx], ch_idx-0.2, ch_idx+0.2, colors='r', 
                      linestyles='--')
            
            # Mark epochs above threshold
            bad_epochs = peak_to_peak[bad_epochs_per_channel[ch_idx], ch_idx]
            if len(bad_epochs) > 0:
                ax1.scatter([ch_idx] * len(bad_epochs), bad_epochs, 
                           color='red', alpha=0.5, s=20)
        
        ax1.set_xticks(range(len(epochs.ch_names)))
        ax1.set_xticklabels(epochs.ch_names, rotation=45, ha='right')
        ax1.set_ylabel('Peak-to-peak amplitude')
        ax1.set_title('Channel-wise distributions with individual thresholds')
        
        # Plot 2: Heatmap of bad epochs
        ax2 = fig.add_subplot(gs[1])
        im = ax2.imshow(bad_epochs_per_channel, aspect='auto', cmap='Reds')
        ax2.set_xlabel('Epoch number')
        ax2.set_yticks(range(len(epochs.ch_names)))
        ax2.set_yticklabels(epochs.ch_names)
        ax2.set_title('Bad epochs per channel (red)')
        plt.colorbar(im)
        
        plt.tight_layout()

        return fig

    def identify_consistently_bad_channels(self, bad_epochs_per_channel, channel_names, 
                                         bad_fraction_threshold):
        """
        Identify channels that are consistently marked as bad across epochs.
    
        A channel is considered consistently bad if the fraction of epochs where it
        exceeds threshold is greater than bad_fraction_threshold.
    
        Parameters
        ----------
        bad_epochs_per_channel : ndarray, shape (n_channels, n_epochs)
            Boolean array marking bad epochs for each channel
        channel_names : list of str
            Names of channels corresponding to rows in bad_epochs_per_channel
        bad_fraction_threshold : float
            Minimum fraction of bad epochs needed to mark a channel as consistently bad (0-1)
    
        Returns
        -------
        bad_channels : ndarray
            Indices of channels marked as consistently bad
    
        Notes
        -----
        Prints a summary of identified bad channels and their respective percentages
        of bad epochs. If no channels exceed the threshold, prints "None".
        """
        bad_fractions = bad_epochs_per_channel.sum(axis=1) / bad_epochs_per_channel.shape[1]
        bad_channels = np.where(bad_fractions >= bad_fraction_threshold)[0]
        
        print(f"\nChannels marked as bad (>{bad_fraction_threshold*100}% bad epochs):")
        if len(bad_channels) > 0:
            for ch_idx in bad_channels:
                print(f"{channel_names[ch_idx]}: {bad_fractions[ch_idx]*100:.1f}% bad epochs")
        else:
            print("None")
        
        return bad_channels

    def identify_bad_epochs(self, bad_epochs_per_channel, bad_channels):
        """
        Identify epochs that should be rejected based on channel quality.
    
        Marks an epoch as bad if it contains any bad channels, after excluding
        consistently bad channels from consideration.
    
        Parameters
        ----------
        bad_epochs_per_channel : ndarray, shape (n_channels, n_epochs)
            Boolean array marking bad epochs for each channel
        bad_channels : array-like
            Indices of channels already marked as consistently bad
    
        Returns
        -------
        bad_epochs : ndarray
            Indices of epochs to be rejected
    
        Notes
        -----
        Excludes consistently bad channels from the rejection criteria to avoid
        double-penalizing epochs. Prints a summary of the epochs marked for rejection.
        """
        # Remove bad channels from consideration
        good_channels = np.ones(len(bad_epochs_per_channel), dtype=bool)
        good_channels[bad_channels] = False
        
        # Find epochs with any bad channels (after excluding consistently bad channels)
        n_bad_per_epoch = bad_epochs_per_channel[good_channels].sum(axis=0)
        bad_epochs = np.where(n_bad_per_epoch > 0)[0]
        
        print(f"\nEpochs to be dropped (any bad channels):")
        if len(bad_epochs) > 0:
            print(f"{len(bad_epochs)} epochs: {sorted(bad_epochs.tolist())}")
        else:
            print("None")
        
        return bad_epochs

    def plot_marked_epochs(self, epochs, bad_epochs_idx, bad_channel_names, bad_epochs_per_channel, n_epochs=20):
        """
        Plot epochs with color coding:
        - Normal data in black
        - Bad channels (marked in all epochs) in grey
        - Channels marked as bad in kept epochs in orange
        - Bad channels in rejected epochs in bold red
        - All signals in rejected epochs in pink
        
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object
        bad_epochs_idx : array-like
            Indices of epochs marked as bad
        bad_channel_names : list
            Names of channels marked as bad in all epochs
        bad_epochs_per_channel : array
            Boolean array marking bad epochs per channel
        n_epochs : int
            Number of epochs to plot
        """
        # Ensure n_epochs isn't larger than total epochs
        n_epochs = min(n_epochs, len(epochs))
        
        # Create color and weight lists for each epoch and channel
        n_channels = len(epochs.ch_names)
        colors = [['k'] * n_channels for _ in range(len(epochs))]  # default black
        linewidths = [[0.5] * n_channels for _ in range(len(epochs))]  # default thin
        
        # First mark everything in bad epochs as pink
        for epoch_idx in bad_epochs_idx:
            colors[epoch_idx] = ['pink'] * n_channels
        
        # Then apply specific channel markings
        for epoch_idx in range(len(epochs)):
            for ch_idx in range(n_channels):
                ch_name = epochs.ch_names[ch_idx]
                
                if ch_name in bad_channel_names:
                    # Consistently bad channels in grey
                    colors[epoch_idx][ch_idx] = 'gray'
                    
                elif bad_epochs_per_channel[ch_idx, epoch_idx]:
                    if epoch_idx in bad_epochs_idx:
                        # Bad channels in rejected epochs in bold red
                        colors[epoch_idx][ch_idx] = 'red'
                        linewidths[epoch_idx][ch_idx] = 2.0  # make bold
                    else:
                        # Bad channels in kept epochs in orange
                        colors[epoch_idx][ch_idx] = 'orange'
        
        # Plot epochs with custom colors
        fig = epochs.plot(n_epochs=n_epochs, epoch_colors=colors, picks='all',
                         event_id=epochs.event_id,
                         title='Epochs (grey=bad channel, orange=bad in kept, red=bad in rejected, pink=rejected)',
                         scalings=None)  # scalings=None to use MNE default scaling
        
        # Apply linewidths to the plot
        #for ax in fig.axes:
        #    for line_idx, line in enumerate(ax.lines):
        #        epoch_idx = line_idx // n_channels
        #        ch_idx = line_idx % n_channels
        #        if epoch_idx < len(linewidths) and ch_idx < len(linewidths[0]):
        #            line.set_linewidth(linewidths[epoch_idx][ch_idx])
        
        return fig


    def print_cleaning_summary(self, epochs_original, epochs_clean, cleaning_method='autoreject', 
                         ar_instance=None, threshold_params=None):
        """Print detailed cleaning summary for epoch rejection and add to report.
        Works with AutoReject, channel threshold, and manual cleaning methods.
        
        Parameters
        ----------
        epochs_original : mne.Epochs
            Original epochs before cleaning
        epochs_clean : mne.Epochs
            Cleaned epochs after rejection
        cleaning_method : str
            Method used for cleaning: 'autoreject', 'threshold', or 'manual'
        ar_instance : autoreject.AutoReject, optional
            The fitted AutoReject instance if using AutoReject
        threshold_params : dict, optional
            Dictionary containing threshold parameters if using threshold method:
            - 'q': quantile threshold
            - 'channel_fraction_threshold': fraction for bad channel marking
            
        Returns
        -------
        str
            Formatted summary text
        """
        # Initialize list to collect summary lines 
        sid = epochs_original.info['subject_info']['sid']
        summary_lines = []
        text_lines = []
        
        def add_line(line=""):
            print(line)
            text_lines.append(line)
            # Convert spaces at start of line to &nbsp; to preserve indentation
            html_line = line
            if html_line.startswith(" "):
                leading_spaces = len(html_line) - len(html_line.lstrip())
                html_line = "&nbsp;" * leading_spaces + html_line.lstrip()
            summary_lines.append(html_line + "<br>")
        
        method_map = {
            'autoreject': 'AUTOREJECT',
            'threshold': 'THRESHOLD-BASED',
            'manual': 'MANUAL'
        }
        method_name = method_map.get(cleaning_method.lower(), cleaning_method.upper())
        add_line(f"\n=== {method_name} CLEANING SUMMARY ===")
        
        # Overall epoch rejection statistics
        n_epochs_total = len(epochs_original)
        n_epochs_kept = len(epochs_clean)
        n_epochs_rejected = n_epochs_total - n_epochs_kept
        
        add_line("\nOverall Epoch Statistics:")
        add_line(f"- Total epochs: {n_epochs_total}")
        add_line(f"- Rejected epochs: {n_epochs_rejected} ({n_epochs_rejected/n_epochs_total*100:.1f}%)")
        add_line(f"- Kept epochs: {n_epochs_kept} ({n_epochs_kept/n_epochs_total*100:.1f}%)")
        
        # Statistics by epoch type
        if hasattr(epochs_original, 'event_id') and epochs_original.event_id:
            add_line("\nStatistics by Epoch Type:")
            for event_name, event_id in epochs_original.event_id.items():
                # Original counts
                orig_events = epochs_original.events[:, 2] == event_id
                n_orig = np.sum(orig_events)
                
                # Kept counts (need to check if event still exists in cleaned epochs)
                if event_id in epochs_clean.event_id:
                    kept_events = epochs_clean.events[:, 2] == event_id
                    n_kept = np.sum(kept_events)
                else:
                    n_kept = 0
                
                n_rejected = n_orig - n_kept
                
                add_line(f"\n{event_name}:")
                add_line(f"- Total: {n_orig}")
                add_line(f"- Kept: {n_kept} ({n_kept/n_orig*100:.1f}%)")
                add_line(f"- Rejected: {n_rejected} ({n_rejected/n_orig*100:.1f}%)")
        
        # Channel Statistics
        orig_bads = epochs_original.info['bads']
        clean_bads = epochs_clean.info['bads']
        
        add_line("\nChannel Statistics:")
        add_line(f"- Total channels: {len(epochs_original.ch_names)}")
        add_line(f"- Originally marked bad: {len(orig_bads)}")
        add_line(f"- Additionally marked bad: {len(set(clean_bads) - set(orig_bads))}")
        add_line(f"- Total bad channels: {len(clean_bads)} "
                f"({len(clean_bads)/len(epochs_original.ch_names)*100:.1f}%)")
        
        if clean_bads:
            add_line("\nBad Channels:")
            for ch in clean_bads:
                if ch in orig_bads:
                    add_line(f"- {ch} (marked bad initially)")
                else:
                    marking_method = {
                        'autoreject': 'during autoreject',
                        'threshold': 'during threshold rejection',
                        'manual': 'during manual inspection'
                    }.get(cleaning_method.lower(), f'during {cleaning_method}')
                    add_line(f"- {ch} ({marking_method})")
        
        # Method-specific parameters
        if cleaning_method == 'autoreject' and ar_instance is not None:
            add_line("\nAutoReject Parameters:")
            add_line(f"- Consensus: {ar_instance.consensus}")
            add_line(f"- n_interpolate: {ar_instance.n_interpolate}")
            add_line(f"- thresh_method: {ar_instance.thresh_method}")
            
            # Get rejection thresholds if available
            if hasattr(ar_instance, 'thresholds_'):
                add_line("\nRejection Thresholds:")
                for ch_idx, thresh in enumerate(ar_instance.thresholds_):
                    ch_name = epochs_original.ch_names[ch_idx]
                    add_line(f"- {ch_name}: {thresh:.2f}")
                    
        elif cleaning_method == 'threshold' and threshold_params is not None:
            add_line("\nThreshold Parameters:")
            add_line(f"- Quantile threshold (q): {threshold_params['q']}")
            add_line(f"- Channel fraction threshold: {threshold_params['channel_fraction_threshold']}")
        
        # Drop log summary
        drop_log = epochs_clean.drop_log
        if drop_log:
            reasons = {}
            for log in drop_log:
                if log:  # if epoch was dropped
                    reason = log[0]  # get the first reason
                    reasons[reason] = reasons.get(reason, 0) + 1
            
            if reasons:
                add_line("\nDrop Reasons:")
                for reason, count in reasons.items():
                    add_line(f"- {reason}: {count} epochs ({count/n_epochs_total*100:.1f}%)")
        
        # Join lines
        text_summary = "\n".join(text_lines)
        html_summary = ''.join(summary_lines)
        
        # Add CSS styles to preserve whitespace and use monospace font
        html_summary = f'<div style="white-space: pre-wrap; font-family: monospace;">{html_summary}</div>'
        
        # Add to report if provided
        if hasattr(self, 'report') and self.report is not None:
            try:
                section_map = {
                    'autoreject': 'AutoReject',
                    'threshold': 'Threshold Rejection',
                    'manual': 'Manual Rejection'
                }
                section_name = section_map.get(cleaning_method.lower(), cleaning_method.title())
                
                self.report.add_html(
                    html=html_summary,
                    title=f"{method_name} Epoch Rejection",
                    section=section_name,
                    tags=['cleaning_summary', cleaning_method.lower(), sid]
                )
                self.report.save(str(self.report_path), overwrite=True, open_browser=False)
    
            except Exception as e:
                print(f"Warning: Failed to add cleaning summary to report: {e}")
        
        return text_summary
    
    def clean_epochs_with_channel_thresholds(self, epochs, q, 
                                           channel_fraction_threshold,
                                           plot_results=True):
        """
        Clean epochs using channel-specific amplitude thresholds.
    
        Implements a two-stage cleaning process:
        1. Identifies consistently bad channels based on their performance across epochs
        2. Rejects individual epochs with any remaining bad channels
    
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object to clean
        q : float
            Quantile threshold for peak-to-peak amplitude (0-1)
        channel_fraction_threshold : float
            Fraction of epochs that must be bad to mark a channel as consistently bad (0-1)
        plot_results : bool, optional
            Whether to generate diagnostic plots, by default True
    
        Returns
        -------
        epochs_clean : mne.Epochs
            Cleaned epochs with bad channels marked and bad epochs removed
        bad_epochs : array-like
            Indices of epochs that were removed
        bad_channel_names : list of str
            Names of channels marked as consistently bad
    
        Notes
        -----
        The cleaning process:
        1. Computes channel-specific amplitude thresholds
        2. Marks channels exceeding threshold in >channel_fraction_threshold epochs
        3. Rejects epochs with any bad channels
        4. Generates diagnostic plots if plot_results=True
        5. Adds results to report if available
        """

        print("\nComputing channel-specific thresholds...")
        thresholds, peak_to_peak, bad_epochs_per_channel = self.compute_channel_specific_thresholds(
            epochs, q=q)
        
        if plot_results:
            fig=self.plot_channel_thresholds(epochs, thresholds, peak_to_peak, bad_epochs_per_channel)
            # Add to report if it exists
            # Add to report if it exists
            sid=epochs.info['subject_info']['sid']
            if hasattr(self, 'report') and self.report is not None:
                try:
                    self.report.add_figure(fig=fig, title=f"Treshold epoch rejection: {sid}", section="Treshold rejection", tags=[sid, "Treshold_Rejection"])
                    self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                except Exception as e:
                    self.logger.warning(f"Failed to add Treshold epoch rejection plot to report for subject {sid}: {e}")
        
        # Identify bad channels
        bad_channels = self.identify_consistently_bad_channels(
            bad_epochs_per_channel, 
            epochs.ch_names,
            channel_fraction_threshold
        )
        
        # Get bad channel names
        bad_channel_names = [epochs.ch_names[i] for i in bad_channels]
        
        # Identify bad epochs (any with bad channels)
        bad_epochs = self.identify_bad_epochs(
            bad_epochs_per_channel,
            bad_channels
        )
        
        # Create cleaned epochs
        epochs_clean = epochs.copy()
        
        # Mark bad channels
        if bad_channel_names:
            epochs_clean.info['bads'] = bad_channel_names
        
        # Drop bad epochs
        if len(bad_epochs) > 0:
            epochs_clean.drop(bad_epochs, reason='bad_channels')
        
        # Print comprehensive summary
        #self.print_cleaning_summary(epochs, epochs_clean, bad_epochs, bad_channel_names,
        #                     bad_epochs_per_channel, q, channel_fraction_threshold)
        
        if plot_results:
            print("\nPlotting epochs:")
            print("- Black: Good data")
            print("- Orange: Channels above threshold in kept epochs")
            print("- Bold red: Bad channels in rejected epochs")
            print("- Pink: All signals in rejected epochs")
            print("- Grey: Bad channels (marked for all epochs)")
            self.plot_marked_epochs(epochs, bad_epochs, bad_channel_names, 
                             bad_epochs_per_channel, n_epochs=20)
        
        return epochs_clean, bad_epochs, bad_channel_names

    def clean_epochs_with_autoreject(self, epochs, plot_results=True):
        """
        Clean epochs using AutoReject with no interpolation.
        
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object to clean
        plot_results : bool, optional
            Whether to generate diagnostic plots, by default True
        
        Returns
        -------
        epochs_clean : mne.Epochs
            Cleaned epochs with bad epochs removed
        bad_epochs : array-like
            Indices of epochs that were removed
        bad_channel_names : list of str
            Names of channels marked as consistently bad
        """
        from autoreject import AutoReject
        import numpy as np
        import matplotlib.pyplot as plt
        
        print("\nInitializing AutoReject...")
        ar = AutoReject(
            n_interpolate=0,  # Disable interpolation
            consensus=None,   # Let AutoReject determine consensus
            thresh_method='random_search',  # Simpler optimization
            verbose=True
        )
        
        # Fit AutoReject
        print("\nFitting AutoReject...")
        ar.fit(epochs)
        
        # Get clean epochs
        epochs_clean = epochs.copy()
        epochs_clean = ar.transform(epochs_clean)
        
        # Identify bad epochs (those that were dropped)
        bad_epochs = np.where([i not in epochs_clean.selection for i in range(len(epochs))])[0]
        
        # Identify bad channels (those that contributed to rejection in >50% of bad epochs)
        bad_channel_mask = np.zeros(len(epochs.ch_names))
        for epoch_idx in bad_epochs:
            # Get rejection log for this epoch
            epoch_data = epochs[epoch_idx].get_data()
            for ch_idx in range(len(epochs.ch_names)):
                if ar.local_reject_.is_bad_channel(epoch_data[ch_idx]):
                    bad_channel_mask[ch_idx] += 1
                    
        bad_channels = np.where(bad_channel_mask / len(bad_epochs) > 0.5)[0]
        bad_channel_names = [epochs.ch_names[i] for i in bad_channels]
        
        # Print summary
        print("\nCleaning Summary:")
        print(f"Total epochs: {len(epochs)}")
        print(f"Epochs removed: {len(bad_epochs)} ({len(bad_epochs)/len(epochs)*100:.1f}%)")
        print(f"Bad channels: {', '.join(bad_channel_names) if bad_channel_names else 'None'}")
        
        if plot_results:
            # Plot 1: Channel rejection frequencies
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(epochs.ch_names)), bad_channel_mask/len(bad_epochs)*100)
            plt.xticks(range(len(epochs.ch_names)), epochs.ch_names, rotation=45)
            plt.ylabel('Rejection frequency (%)')
            plt.title('Channel Rejection Frequencies')
            
            # Add to report if it exists
            if hasattr(self, 'report') and self.report is not None:
                try:
                    sid = epochs.info['subject_info']['sid']
                    self.report.add_figure(
                        fig=plt.gcf(),
                        title=f"AutoReject channel statistics: {sid}",
                        section="AutoReject",
                        tags=[sid, "AutoReject_Stats"]
                    )
                    self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                except Exception as e:
                    print(f"Failed to add plot to report: {e}")
            
            # Plot 2: Compare epochs before and after
            n_epochs_to_plot = min(20, len(epochs))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Before cleaning
            epochs.plot(n_epochs=n_epochs_to_plot, picks='eeg', axes=ax1)
            ax1.set_title('Before AutoReject')
            
            # After cleaning
            epochs_clean.plot(n_epochs=n_epochs_to_plot, picks='eeg', axes=ax2)
            ax2.set_title('After AutoReject')
            
            # Add to report
            if hasattr(self, 'report') and self.report is not None:
                try:
                    self.report.add_figure(
                        fig=plt.gcf(),
                        title=f"AutoReject epochs comparison: {sid}",
                        section="AutoReject",
                        tags=[sid, "AutoReject_Epochs"]
                    )
                    self.report.save(str(self.report_path), overwrite=True, open_browser=False)
                except Exception as e:
                    print(f"Failed to add plot to report: {e}")
        
        return epochs_clean, bad_epochs, bad_channel_names
    
    def manage_bad_channels(self, raw, interpolate=False):
        """
        Interactive command-line interface for manually marking bad channels.
    
        Allows users to input channel names to be marked as bad, with validation
        against available channels and confirmation before applying changes.
    
        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance to mark bad channels in
        interpolate : bool, optional
            Whether to interpolate bad channels, by default False
            (Note: This parameter is currently unused but prepared for future implementation)
    
        Returns
        -------
        raw : mne.io.Raw
            Raw instance with updated bad channels list
    
        Notes
        -----
        Features:
        - Lists available channel types and total channel count
        - Validates input channel names against raw.info['ch_names']
        - Shows similar channel names for typos
        - Requires confirmation before applying changes
        - Allows early exit with 'quit' or 'none' commands
        - Supports multiple channel entries with double-Enter to finish
        """
        print("\nEnter bad channel names (one per line).")
        print("Press Enter twice (empty line) when done.")
        print("Type 'quit' or 'none' to exit without marking channels.")
        print("Available channel types:", set(raw.get_channel_types()))
        print(f"Total channels: {len(raw.ch_names)}")
        
        bad_channels = []
        all_channels = raw.ch_names
        
        while True:
            channel = input("Enter channel name (or 'quit'/'none' to exit): ").strip().lower()
            
            # # Check if user wants to quit
            # if channel in ['quit', 'none']:
            #     print("Exiting without marking channels.")
            #     return raw
            
            # Check if user is done (empty line)
            if not channel:
                break  # Allow empty line to finish, even with no channels
                    
            # Validate channel name
            if channel in all_channels:
                if channel in bad_channels:
                    print(f"Channel '{channel}' already marked as bad.")
                else:
                    bad_channels.append(channel)
                    print(f"Added '{channel}' to bad channels.")
            else:
                print(f"Error: '{channel}' not found in channel list.")
                print("Available channels starting with same letter:")
                # Show similar channel names to help with typos
                similar_channels = [ch for ch in all_channels if ch.startswith(channel[0])] if channel else []
                if similar_channels:
                    print(', '.join(similar_channels))
        
        # Final summary
        if bad_channels:
            print("\nMarked as bad:", len(bad_channels), "channels:")
            print(', '.join(bad_channels))
            
            # Confirm with user
            confirm = input("\nConfirm marking these channels as bad? (y/n): ").lower()
            if confirm != 'y':
                print("Cancelled. No channels marked as bad.")
            else:
                raw.info['bads'].extend(bad_channels)
                print("Updated raw.info['bads']:", raw.info['bads'])
                
            
        else:
            print("\nNo channels marked as bad.")
        
        return raw
    
    
  
    def plot_channel_variance(self, raw):
        """
        Plot channel signal variance with color-coding by channel type.
    
        Creates a bar plot showing variance of each channel's signal, with bars
        color-coded to distinguish MEG channels, reference channels, and bad channels.
    
        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance to analyze
    
        Notes
        -----
        Plot features:
        - Blue bars: Regular MEG channels
        - Grey bars: Reference MEG channels
        - Red bars: Channels marked as bad
        - Includes legend for channel types
        - Vertical channel name labels
        - Automatically adds plot to report if report_path is set
    
        The plot is useful for identifying channels with unusual variance that
        might indicate technical issues or artifacts.
        """
        # Load the raw data file
        _raw = raw.copy()
        # Select only 'meg' and 'ref_meg' channels
        picks = mne.pick_types(_raw.info, meg=True, ref_meg=True, exclude=[])
        selected_ch_names = [_raw.ch_names[i] for i in picks]
        # Calculate variance for selected channels
        data = raw.get_data(picks=picks)
        variances = np.var(data, axis=1)
        # Create a bar plot
        fig = plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(variances)), variances)
        plt.xlabel('Channel')
        plt.ylabel('Variance')
        plt.title('Channel variance (meg and ref_meg)')
        # Set y-axis limit to 1e-19
        # plt.ylim(0, 1e-18)
        # Add channel names to x-axis
        plt.xticks(range(len(variances)), selected_ch_names, rotation='vertical')
        # Color-code bars based on channel type and bad channels
        meg_color = 'blue'
        ref_meg_color = 'grey'
        bad_color = 'red'
        for i, bar in enumerate(bars):
            ch_name = selected_ch_names[i]
            chn_t = str(_raw.info['chs'][picks[i]]['kind'])
            # Check if channel is marked as bad
            if ch_name in _raw.info['bads']:
                bar.set_color(bad_color)
            else:
                if self.check_meg_channel_type(chn_t) == 'MEG_CH':
                    bar.set_color(meg_color)
                elif self.check_meg_channel_type(chn_t) == 'REF_MEG_CH':
                    bar.set_color(ref_meg_color)
                else:
                    bar.set_color('white')
        # Add a legend with bad channels included
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=meg_color, edgecolor='black', label='meg'),
            Patch(facecolor=ref_meg_color, edgecolor='black', label='ref_meg'),
            Patch(facecolor=bad_color, edgecolor='black', label='known bad chns')
        ]
        # Extract subject ID
        sid = re.search(r'sub-([A-Za-z0-9]{3})', raw.filenames[0]).group(1)
        plt.legend(handles=legend_elements, loc='upper right')    
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        # Add to report if it exists
        if hasattr(self, 'report') and self.report is not None:
            try:
                self.report.add_figure(fig=fig, title=f"Channels variance: {sid}", section="Channels variance", tags=[sid, "Chns_variance"])
            except Exception as e:
                self.logger.warning(f"Failed to add channels variance to report for subject {sid}: {e}")

        
        
    def plot_infH_refH_loc(self, raw):
        """
        Visualize MEG sensor locations in both 3D space and 2D topographic projection.
    
        Creates a figure with two subplots:
        1. 3D scatter plot of sensor positions with channel labels
        2. 2D topographic projection of sensor layout
    
        Parameters
        ----------
        raw : mne.io.Raw
            Raw MEG data instance containing channel location information
    
        Returns
        -------
        raw : mne.io.Raw
            The input raw instance (after dropping 'di32' channel)
    
        Notes
        -----
        Plot features:
        - 3D plot:
            * Cubic aspect ratio
            * Channel labels using first 3 letters
            * Equal axis limits and tick spacing
        - 2D plot:
            * Topographic projection using MNE's plot_sensors
            * Channel names and positions
        - Automatically adds plots to report if report_path is set
        - Also generates channel variance plot via plot_ch_var()
    
        See Also
        --------
        plot_ch_var : For additional channel variance visualization
        """
        _raw = raw.copy()
        _raw.drop_channels(['di32'])
        
        # Create a new figure with two subplots side by side
        fig = plt.figure(figsize=(20, 10))
        
        # 3D plot (left subplot)
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Extract sensor positions 
        sensor_locs = _raw.info['chs']
        x = [ch['loc'][0] for ch in sensor_locs]
        y = [ch['loc'][1] for ch in sensor_locs]
        z = [ch['loc'][2] for ch in sensor_locs]
        
        # First 3 letters of channel names
        labels = [name[:3] for name in _raw.info['ch_names']]
        
        # Create the 3D plot
        sc = ax1.scatter(x, y, z, c='r', marker='o')
        
        # Add labels
        for i in range(len(x)):
            ax1.text(x[i], y[i], z[i], labels[i], fontsize=7, ha='right')
        
        # Set axis limits and ticks
        min_val = min(min(x), min(y), min(z))
        max_val = max(max(x), max(y), max(z))
        ticks = np.linspace(min_val, max_val, num=5)
        ax1.set_xlim([min_val, max_val])
        ax1.set_ylim([min_val, max_val])
        ax1.set_zlim([min_val, max_val])
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.set_zticks(ticks)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Make the plot cubic
        ax1.set_box_aspect((1, 1, 1))
        ax1.set_title('Channels in 3D space', fontsize=16)
        
        # 2D plot (right subplot)
        ax2 = fig.add_subplot(122)
        
        mp_nr = next((subj['micro_pilot_nr'] for subj in self.subj_info['subjs']), None)
        if "adult" not in mp_nr:
            print("Aligning infant helmet")
            raw = self.align_topo_locs(_raw)
            
        mne.viz.plot_sensors(_raw.info, 
                             kind='topomap', 
                             ch_type='mag', 
                             show_names=True, 
                             ch_groups=None, 
                             to_sphere=True, 
                             axes=ax2, 
                             show=False, 
                             sphere=None, 
                             pointsize=None, 
                             linewidth=2, 
                             cmap=None,
                             verbose=None)
        ax2.set_title('Channels topomap', fontsize=16)
        
        # Add overall title to the figure
        sid = re.search(r'sub-([A-Za-z0-9]{3})', _raw.filenames[0]).group(1)
        fig.suptitle(f"Channel Locations for {sid}", fontsize=20)
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Add to report if it exists
        if hasattr(self, 'report') and self.report is not None:
            try:
                self.report.add_figure(fig=fig, title=f"Channel Locations: {sid}", section="Channel Locations", tags=[sid, "Chans_Loc"])
            except Exception as e:
                self.logger.warning(f"Failed to add channel loc figures to report for subject {sid}: {e}")
        
        self.plot_ch_var(_raw)
        
        del _raw
        return raw
    
    
    def convert_lists_to_numpy(self, ref_loc):
        """
        Convert specific list fields in reference location dictionary to numpy arrays.
    
        Recursively processes dictionary to convert geometric properties from
        lists to numpy arrays for computational efficiency.
    
        Parameters
        ----------
        ref_loc : dict
            Dictionary containing reference location information
    
        Returns
        -------
        dict
            Processed dictionary with lists converted to numpy arrays
    
        Notes
        -----
        Converts the following fields if present:
        - position
        - unit_vector_x
        - unit_vector_y
        - unit_vector_z
    
        The function recursively processes nested dictionaries to ensure all
        relevant fields are converted.
        """
        if isinstance(ref_loc, dict):
            for key, value in ref_loc.items():
                if isinstance(value, dict):
                    ref_loc[key] = self.convert_lists_to_numpy(value)
                elif isinstance(value, list) and key in ['position', 'unit_vector_x', 'unit_vector_y', 'unit_vector_z']:
                    ref_loc[key] = np.array(value)
        return ref_loc
    
    