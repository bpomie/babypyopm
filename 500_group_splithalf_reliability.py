import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from joblib import Parallel, delayed

datadir = '/rds/projects/o/oriolig-babyopm/project_setup_methods/data/'
preproc = 'processed_2_filter_ica'
fname = 'sub-{subj}_file-oddballTones_processed_2_filter_ica.fif'
fulltemplate = os.path.join(datadir, 'sub-{subj}', preproc, fname)

cname = 'sub-{subj}_badchannels.tsv'
chantemplate = os.path.join(datadir, 'sub-{subj}', cname)
oname = 'sub-{subj}_epochs_bad.txt'
epochtemplate = os.path.join(datadir, 'sub-{subj}', oname)


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


subjs = np.delete(np.arange(101, 108), 2)
R = {}
corrR = {}
for ii in range(len(subjs)):
    print('-'*25)

    fullfile = fulltemplate.format(subj=str(subjs[ii]))
    raw = mne.io.read_raw(fullfile, preload=True)

    bad_chans = pd.read_csv(chantemplate.format(subj=str(subjs[ii])), sep="\t")
    raw.info['bads'] = list(bad_chans['badchannelslots'].values)
    print(raw.info['bads'])

    bad_epochs = np.loadtxt(epochtemplate.format(subj=str(subjs[ii])))

    # Quick extra filter to remove drift
    raw.filter(l_freq=1, h_freq=45, method='iir', iir_params={'order': 5, 'ftype': 'butter'})

    events = mne.find_events(raw)
    event_dict = {"infreq/tone/low": 2,
                  "infreq/tone/high": 4,
                  "freq/tone/low": 8,
                  "freq/tone/high": 12}

    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=0.6)
    epochs.drop(bad_epochs)
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
plt.savefig('reliability_figure.png', dpi=300, bbox_inches='tight')



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
plt.savefig('reliability_figure_withcorrection.png', dpi=300, bbox_inches='tight')
