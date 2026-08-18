import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from joblib import Parallel, delayed

# =============================================================================
# INDICATE YOUR PATH
# =============================================================================

# Insert the path to your project folder
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled/'

datadir = os.path.join(root_data_path, 'data')

preproc = 'processed_2_filter_ica'
fname = '{subj}_file-oddballTones_preprocessing_routine_3.fif'
fulltemplate = os.path.join(datadir, '{subj}', preproc, fname)

cname = '{subj}_badchannels.tsv'
chantemplate = os.path.join(datadir, '{subj}', cname)
oname = '{subj}_epochs_bad.txt'
epochtemplate = os.path.join(datadir, '{subj}', oname)

# Results path
path_results_reliability = os.path.join(root_data_path, 'results', 'reliability')
os.makedirs(path_results_reliability, exist_ok=True)


def compute_split_half(iboot, isubsample):
    inds = np.random.permutation(n_trials)
    n = subsamples[isubsample]
    half = n // 2

    if n > freq.shape[0]:
        return iboot, isubsample, np.nan, np.nan

    A = freq[inds[:n][:half], :, :].mean(axis=0).reshape(-1)
    B = freq[inds[:n][half:], :, :].mean(axis=0).reshape(-1)

    r = np.corrcoef(A, B)[0, 1]

    # Spearman-Brown correction
    corr_r = (2 * r) / (1 + r)

    return iboot, isubsample, r, corr_r


# =============================================================================
# PARTICIPANT LIST
# =============================================================================

# List all subjects in the data folder
subjs = sorted([f for f in os.listdir(datadir)
                if os.path.isdir(os.path.join(datadir, f)) and f.startswith('sub-')])
print(f"Found {len(subjs)} subjects: {subjs}")

# Alternatively, manually specify subjects:
# subjs = ['sub-101', 'sub-102', 'sub-107']

R = {}
corrR = {}
for ii in range(len(subjs)):
    print('-'*25)
    print(subjs[ii])

    fullfile = fulltemplate.format(subj=subjs[ii])

    if not os.path.exists(fullfile):
        print(f"Skipping {subjs[ii]}: data file not found")
        continue

    raw = mne.io.read_raw(fullfile, preload=True)

    bad_chans = pd.read_csv(chantemplate.format(subj=subjs[ii]), sep="\t")
    bad_chans = bad_chans['badchannelslots'].tolist()

    # Clean up entries (drop NaNs, strip whitespace)
    bad_chans = [str(ch).strip() for ch in bad_chans if pd.notna(ch)]

    # Keep only bad channels that actually exist in this recording
    # (guards against naming-convention mismatches, e.g. 's24_bz' vs 'L11')
    missing_bads = [ch for ch in bad_chans if ch not in raw.ch_names]
    bad_chans = [ch for ch in bad_chans if ch in raw.ch_names]

    if missing_bads:
        print(f"WARNING: bad channels not found in data for {subjs[ii]}, skipped: {missing_bads}")

    raw.info['bads'] = bad_chans
    print(raw.info['bads'])

    # Load bad epochs (optional file; may be absent or empty)
    epochfile = epochtemplate.format(subj=subjs[ii])
    if os.path.exists(epochfile):
        bad_epochs = np.atleast_1d(np.loadtxt(epochfile))
    else:
        print(f"No bad epochs file for {subjs[ii]}")
        bad_epochs = np.array([])

    # Quick extra filter to remove drift
    raw.filter(l_freq=1, h_freq=45, method='iir', iir_params={'order': 5, 'ftype': 'butter'})

    events = mne.find_events(raw)
    event_dict = {"infreq/tone/low": 2,
                  "infreq/tone/high": 4,
                  "freq/tone/low": 8,
                  "freq/tone/high": 12}

    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=0.6)
    epochs.load_data()

    if len(bad_epochs) > 0:
        bad_epochs = bad_epochs.astype(int)
        # Filter out indices that are out of range
        valid_bad_epochs = bad_epochs[bad_epochs < len(epochs)]
        if len(valid_bad_epochs) < len(bad_epochs):
            print(f"  (Skipped {len(bad_epochs) - len(valid_bad_epochs)} out-of-range indices)")
        if len(valid_bad_epochs) > 0:
            print(f"Dropping {len(valid_bad_epochs)} bad epochs")
            epochs.drop(valid_bad_epochs)

    epochs.apply_baseline()

    freq = epochs['freq'].get_data(picks='meg')
    infreq = epochs['infreq'].get_data(picks='meg')

    # ------------

    subsamples = np.r_[np.arange(10, 400, 10)]
    nbootstraps = 500
    nprocesses = -1

    n_trials = freq.shape[0]
    half = n_trials // 2

    # Run in parallel
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_split_half)(iboot, isubsample)
        for isubsample in range(len(subsamples))
        for iboot in range(nbootstraps)
    )

    # Fill result array
    Ri = np.zeros((nbootstraps, len(subsamples)))
    Rj = np.zeros((nbootstraps, len(subsamples)))

    #for iboot, isubsample, r, corr_r in results:
    for blah in results:
        iboot, isubsample, r, corr_r = blah
        Ri[iboot, isubsample] = r
        Rj[iboot, isubsample] = corr_r
    R[str(subjs[ii])] = Ri
    corrR[str(subjs[ii])] = Rj


plt.figure(figsize=(12, 8))
for key, val in R.items():
    plt.plot(subsamples, val.mean(axis=0), label=key)
plt.xticks(subsamples[::3])
plt.plot(subsamples, np.ones_like(subsamples)*0.6, 'k--')
plt.plot(subsamples, np.ones_like(subsamples)*0.8, 'k:')
for tag in ['top','right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Number of trials', fontsize=14)
plt.ylabel('Split Half Reliability', fontsize=14)
plt.legend(bbox_to_anchor=(1, 1.1), fontsize=10, framealpha=0.9)
plt.title('Bootstrapped split-half reliability across number of trials\nFrequent condition across all trials and timepoints', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(path_results_reliability, 'reliability_figure.png'), dpi=300, bbox_inches='tight')



plt.figure(figsize=(12, 8))
for key, val in corrR.items():
    plt.plot(subsamples, val.mean(axis=0), label=key)
plt.xticks(subsamples[::3])
plt.plot(subsamples, np.ones_like(subsamples)*0.6, 'k--')
plt.plot(subsamples, np.ones_like(subsamples)*0.8, 'k:')
for tag in ['top','right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Number of trials', fontsize=14)
plt.ylabel('Split Half Reliability', fontsize=14)
plt.legend(bbox_to_anchor=(1, 1.1), fontsize=10, framealpha=0.9)
plt.title('Bootstrapped split-half reliability across number of trials\nFrequent condition across all trials and timepoints', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(path_results_reliability, 'reliability_figure_withcorrection.png'), dpi=300, bbox_inches='tight')