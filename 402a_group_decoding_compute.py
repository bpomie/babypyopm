#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
402a_group_decoding_compute.py

COMPUTE ONLY - Run this once to generate results.
Then use 402b_group_decoding_figures.py to iterate on visualizations.

@author: a.pesquita@bham.ac.uk
"""

import numpy as np
np.alltrue = np.all

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from scipy.signal import find_peaks
import os
import pandas as pd
import json

import utils_study
study = utils_study.Study
utils = utils_study.Utils()

# =============================================================================
# PARAMETERS
# =============================================================================

input_folder = 'processed_2_filter_ica'
task = 'oddballTones'

root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

# =============================================================================
# SET UP
# =============================================================================

paths = utils.get_paths(root_data_path)

path_results_decoding = os.path.join(
    root_data_path, 'results', 'preprocessing_routine_2', 'decoding_group'
)
os.makedirs(path_results_decoding, exist_ok=True)

path_results_peak_stats = os.path.join(
    root_data_path, 'results', 'decoding_peak_stats'
)
os.makedirs(path_results_peak_stats, exist_ok=True)

subjects = [f for f in os.listdir(paths.data) 
            if os.path.isdir(os.path.join(paths.data, f)) and f.startswith('sub-')]
subjects.sort()

print(f"Found {len(subjects)} subjects: {subjects}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_participant(subj, paths):
    print(f"\n{'='*60}")
    print(f"Processing {subj}")
    print('='*60)
    
    path_task_data = os.path.join(
        paths.data, subj, input_folder, f"{subj}_file-oddballTones_processed_2_filter_ica.fif"
    )
    path_bad_channels = os.path.join(paths.data, subj, f"{subj}_badchannels.tsv")
    path_event_dict = os.path.join(paths.data, subj, f"{subj}_event_dict.json")
    
    if not os.path.exists(path_task_data):
        print(f"WARNING: Data file not found for {subj}")
        return None, None
    
    try:
        raw = mne.io.read_raw_fif(path_task_data, preload=True, verbose=False)
        
        if os.path.exists(path_bad_channels):
            bad_channels = pd.read_csv(path_bad_channels, sep='\t')
            bad_channels = bad_channels['badchannelslots'].tolist()
            raw.info['bads'].clear()
            raw.info['bads'].extend(bad_channels)
            print(f"Bad channels: {bad_channels}")
        
        events = mne.find_events(raw, stim_channel='di32', verbose=False)
        
        with open(path_event_dict, 'r') as f:
            event_dict = json.load(f)
        
        epochs = mne.Epochs(
            raw, events, event_id=event_dict,
            tmin=-0.1, tmax=0.5, baseline=None, detrend=1,
            reject_by_annotation=False, preload=True, verbose=False
        )
        
        epochs_freq = epochs["freq/tone"]
        epochs_infreq = epochs["infreq/tone"]
        
        print(f"Frequent: {len(epochs_freq)} epochs")
        print(f"Infrequent: {len(epochs_infreq)} epochs")
        
        min_len = min(len(epochs_freq), len(epochs_infreq))
        
        if min_len < 20:
            print(f"WARNING: Too few epochs ({min_len}) for {subj}")
            return None, None
        
        np.random.seed(42)
        idx_freq = np.random.choice(len(epochs_freq), min_len, replace=False)
        idx_infreq = np.random.choice(len(epochs_infreq), min_len, replace=False)
        
        epochs_equalized = mne.concatenate_epochs([
            epochs_freq[idx_freq], epochs_infreq[idx_infreq]
        ])
        
        print(f"Equalized to {min_len} epochs per condition")
        
        X = epochs_equalized.get_data(picks='mag')
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
        
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear', random_state=42)
        )
        
        time_decoder = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)
        times = epochs_equalized.times
        
        print(f"Mean AUC: {np.mean(scores):.3f}")
        print(f"Peak AUC: {np.max(np.mean(scores, axis=0)):.3f}")
        
        return scores, times
        
    except Exception as e:
        print(f"ERROR processing {subj}: {str(e)}")
        return None, None


def identify_peak(mean_scores, times, min_time=0.0, prominence=0.01, distance=20):
    """Identify the largest peak in the mean decoding curve."""
    post_stim_mask = times >= min_time
    post_stim_scores = mean_scores.copy()
    post_stim_scores[~post_stim_mask] = 0
    
    peak_indices, _ = find_peaks(post_stim_scores, prominence=prominence, distance=distance)
    
    if len(peak_indices) > 1:
        largest_idx = np.argmax(mean_scores[peak_indices])
        peak_indices = np.array([peak_indices[largest_idx]])
    
    return peak_indices[0] if len(peak_indices) > 0 else np.argmax(mean_scores[post_stim_mask])


def run_binomial_test(scores_at_peak, chance_level=0.5):
    n_subjects = len(scores_at_peak)
    n_above_chance = np.sum(scores_at_peak > chance_level)
    result = binomtest(n_above_chance, n_subjects, p=0.5, alternative='greater')
    
    return {
        'n_subjects': n_subjects,
        'n_above_chance': n_above_chance,
        'proportion_above': n_above_chance / n_subjects,
        'expected_by_chance': n_subjects * 0.5,
        'p_value': result.pvalue,
        'mean_auc': np.mean(scores_at_peak),
        'std_auc': np.std(scores_at_peak),
    }


# =============================================================================
# PART 1: PROCESS ALL PARTICIPANTS
# =============================================================================

print("="*70)
print("PART 1: GROUP TEMPORAL DECODING")
print("="*70)

all_scores = []
all_times = []
successful_subjects = []

for subj in subjects:
    scores, times = decode_participant(subj, paths)
    if scores is not None:
        all_scores.append(np.mean(scores, axis=0))
        all_times.append(times)
        successful_subjects.append(subj)

print(f"\n{'='*60}")
print(f"Successfully processed {len(successful_subjects)}/{len(subjects)} subjects")
print('='*60)

if len(all_scores) == 0:
    print("No subjects successfully processed!")
    exit()

# Verify time consistency
time_lengths = [len(t) for t in all_times]
if len(set(time_lengths)) > 1:
    print("WARNING: Different time vectors across subjects!")
    min_time_len = min(time_lengths)
    all_scores = [s[:min_time_len] for s in all_scores]
    all_times = [t[:min_time_len] for t in all_times]

times = all_times[0]
scores_array = np.array(all_scores)

# Compute group statistics
mean_scores_group = np.mean(scores_array, axis=0)
sem_scores_group = np.std(scores_array, axis=0) / np.sqrt(len(all_scores))

# =============================================================================
# PART 2: PEAK STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("PART 2: PEAK STATISTICAL ANALYSIS")
print("="*70)

peak_idx = identify_peak(mean_scores_group, times, min_time=0.0, prominence=0.01, distance=15)
peak_time = times[peak_idx]
peak_value = mean_scores_group[peak_idx]
scores_at_peak = scores_array[:, peak_idx]

print(f"\nPeak: time = {peak_time*1000:.1f} ms, AUC = {peak_value:.3f}")

# Binomial test
binomial_result = run_binomial_test(scores_at_peak)

print(f"\nBinomial test:")
print(f"  Subjects above 0.5: {binomial_result['n_above_chance']}/{binomial_result['n_subjects']}")
print(f"  p-value: {binomial_result['p_value']:.4f}")
if binomial_result['p_value'] < 0.05:
    print(f"  --> SIGNIFICANT at α = 0.05")
else:
    print(f"  --> Not significant at α = 0.05")

# =============================================================================
# SAVE ALL RESULTS
# =============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save comprehensive results for figure script
results_for_figures = {
    'times': times,
    'scores_array': scores_array,
    'mean_scores_group': mean_scores_group,
    'sem_scores_group': sem_scores_group,
    'successful_subjects': successful_subjects,
    'peak_idx': peak_idx,
    'peak_time': peak_time,
    'peak_value': peak_value,
    'scores_at_peak': scores_at_peak,
    'binomial_result': binomial_result,
}

np.save(os.path.join(path_results_peak_stats, 'decoding_results_for_figures.npy'),
        results_for_figures, allow_pickle=True)

# Also save CSV for easy inspection
results_df = pd.DataFrame({
    'time': times,
    'mean_auc': mean_scores_group,
    'sem_auc': sem_scores_group
})
for i, subj in enumerate(successful_subjects):
    results_df[subj] = scores_array[i]

results_df.to_csv(os.path.join(path_results_decoding, 'group_decoding_results.csv'), index=False)

# Save peak summary
summary_df = pd.DataFrame({
    'subject': successful_subjects,
    'auc_at_peak': scores_at_peak,
    'above_chance': scores_at_peak > 0.5
})
summary_df.to_csv(os.path.join(path_results_peak_stats, 'peak_subject_scores.csv'), index=False)

print(f"\nSaved results to:")
print(f"  {os.path.join(path_results_peak_stats, 'decoding_results_for_figures.npy')}")
print(f"  {os.path.join(path_results_decoding, 'group_decoding_results.csv')}")
print(f"  {os.path.join(path_results_peak_stats, 'peak_subject_scores.csv')}")

print("\n" + "="*70)
print("COMPUTE COMPLETE - Now run 402b_group_decoding_figures.py")
print("="*70)
