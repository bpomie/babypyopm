# Temporal Decoding Analysis Scripts

## Overview

These scripts perform time-resolved classification to determine when the brain distinguishes between **frequent vs infrequent tones** in your oddball paradigm. They use a sliding-window logistic regression approach to decode condition labels from MEG sensor data at each time point.

---

## Scripts

### 1. `300_temporal_decoding_freq_vs_infreq.py`
**Purpose**: Single participant temporal decoding analysis

**What it does**:
- Loads preprocessed MEG data for one participant
- Creates epochs for frequent and infrequent tones
- Equalizes trial counts between conditions
- Trains a logistic regression classifier at each time point (-100ms to 500ms)
- Uses 5-fold cross-validation to estimate classification performance
- Generates temporal decoding curve (AUC over time)
- Computes temporal generalization matrix (train at time X, test at time Y)

**Key outputs**:
```
results/preprocessing_routine_2/decoding/
├── {subj}_temporal_decoding.png              # AUC over time plot
├── {subj}_temporal_generalization.png        # Train/test time matrix
└── {subj}_decoding_results.npy               # All results for later analysis
```

---

### 2. `301_group_temporal_decoding.py`
**Purpose**: Group-level temporal decoding across all participants

**What it does**:
- Automatically processes all subjects in your data directory
- Runs temporal decoding for each participant
- Computes group statistics (mean ± SEM across subjects)
- Creates publication-quality group plots
- Saves results in both .npy and .csv formats

**Key outputs**:
```
results/preprocessing_routine_2/decoding_group/
├── group_temporal_decoding.png               # Grand average plot
├── group_decoding_results.npy                # All data for analysis
└── group_decoding_results.csv                # Easy-to-read table format
```

---

## How to Use

### Step 1: Single Participant Analysis

```python
# Edit the script to set your participant
subj = 'sub-107'  # Change this line

# Run the script
python 300_temporal_decoding_freq_vs_infreq.py
```

**What to check**:
- Console output shows number of epochs per condition
- Peak AUC and timing statistics
- Figures display automatically

**Expected runtime**: ~2-5 minutes per participant

---

### Step 2: Group Analysis

```python
# No editing needed - automatically finds all subjects
python 301_group_temporal_decoding.py
```

**What to check**:
- Console shows which subjects were successfully processed
- Any warnings about missing files or insufficient epochs
- Group statistics summary at the end

**Expected runtime**: ~5-15 minutes depending on number of subjects

---

## Method Details

### Data Preprocessing Requirements

**Required files per participant**:
```
data/{subj}/preprocessing_routine_2/{subj}_manual_clean.fif
data/{subj}/{subj}_badchannels.tsv
data/{subj}/{subj}_event_dict.json
```

### Analysis Parameters

**Epochs**:
- Time window: -0.1 to 0.5 seconds
- No baseline correction (classifier learns from raw data)
- Detrending: Linear (order 1)
- Equal number of trials per condition (randomly downsampled)

**Classifier**:
- Algorithm: Logistic Regression
- Regularization: L2 (default)
- Preprocessing: StandardScaler (z-score normalization per channel)
- Cross-validation: 5-fold stratified

**Performance Metric**:
- AUC (Area Under ROC Curve)
- Range: 0.5 (chance) to 1.0 (perfect)
- Values > 0.55 typically indicate significant decoding

---

## Interpreting Results

### Temporal Decoding Curve

The main plot shows **AUC over time**:

```
    AUC
    0.7 |           ╱‾‾‾╲
        |          ╱     ╲
    0.6 |        ╱        ╲___
        |      ╱               
    0.5 |____╱________________  ← Chance level
        |
        -100    0    200   400   Time (ms)
              ↑
         Stimulus onset
```

**Key features to look for**:
- **Peak time**: When does decoding peak? (typically 200-400ms for oddball)
- **Onset time**: When does AUC first exceed threshold? (e.g., 0.55)
- **Duration**: How long does significant decoding last?
- **Pre-stimulus**: Is AUC at chance before stimulus? (should be ~0.5)

### Temporal Generalization Matrix

Shows whether patterns learned at one time generalize to other times:

```
Testing Time
    ↓
    500|        ■■
       |      ■■■■
    300|    ■■■■■■
       |  ■■■■■■
    100|■■■■■
       |
     0 |■
       |_______________
        0  100 300 500 → Training Time
```

**Interpretation**:
- **Diagonal**: Strong diagonal = transient patterns (different at each time)
- **Square block**: Pattern is stable across time window
- **Off-diagonal**: Early pattern generalizes to late times (or vice versa)

### Group Results

**Individual traces (gray)** show variability across participants
**Grand average (blue)** shows consistent group-level effect
**Shaded region** shows standard error of the mean (SEM)

**What to report**:
- Peak AUC and timing: "Peak decoding (AUC = 0.64 ± 0.03) at 320ms"
- Onset: "Significant decoding from 180ms post-stimulus"
- Statistics: "Group mean significantly above chance (p < 0.05, permutation test)"

---

## Common Issues & Solutions

### Issue 1: Too Few Epochs
```
WARNING: Too few epochs (15) for sub-101
```
**Solution**: Check your preprocessing - need at least 20 epochs per condition

### Issue 2: Memory Error
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Close other applications
- Reduce number of cross-validation folds (change `n_splits=5` to `n_splits=3`)
- Set `n_jobs=1` instead of `-1`

### Issue 3: Decoding at Chance
```
Mean AUC: 0.501
```
**Possible causes**:
- Not enough trials (need >30 per condition)
- Data quality issues (check preprocessing)
- Conditions truly not distinguishable in sensor space

### Issue 4: Missing Files
```
WARNING: Data file not found for sub-103
```
**Solution**: Script continues with other subjects - check if this subject exists and has required files

---

## Customization Options

### Change Cross-Validation
```python
# In either script, find this line:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Change to 10-fold CV:
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

### Change Performance Metric
```python
# In the SlidingEstimator call:
time_decoder = SlidingEstimator(
    clf,
    n_jobs=1,
    scoring='accuracy',  # Change from 'roc_auc' to 'accuracy'
    verbose=True
)
```

### Add Different Classifier
```python
# Replace LogisticRegression with SVM:
from sklearn.svm import SVC

clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', random_state=42)
)
```

### Select Specific Sensors
```python
# Before creating epochs, pick specific channels:
sensor_list = ['L01', 'L02', 'R01', 'R02']  # Your sensor names
raw = raw.pick_channels(sensor_list)
```

### Change Time Window
```python
# In the Epochs call:
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,    # Earlier start
    tmax=0.6,     # Later end (be careful of 500ms ISI!)
    baseline=None,
    detrend=1,
    reject_by_annotation=True,
    preload=True
)
```

---

## Statistical Testing (Next Steps)

The scripts currently show **descriptive statistics**. For inference, you should add:

### 1. Permutation Testing
Test whether decoding significantly exceeds chance:
```python
from mne.stats import permutation_cluster_1samp_test

# Subtract chance level (0.5 for AUC)
scores_centered = scores_array - 0.5

# Cluster-based permutation test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    scores_centered,
    n_permutations=10000,
    tail=1,  # One-tailed (above chance)
    threshold=dict(start=0, step=0.2)
)
```

### 2. Condition Contrasts
Compare different experimental manipulations:
```python
# Decode freq vs infreq separately for high/low tones
# Then compare peak times or AUC magnitudes
```

### 3. Correlation with Behavior
```python
# Correlate peak decoding with behavioral measures
from scipy.stats import spearmanr

behavioral_scores = [...]  # Load from file
peak_aucs = np.max(scores_array, axis=1)

r, p = spearmanr(peak_aucs, behavioral_scores)
```

---

## Expected Results

Based on typical oddball paradigms, you should see:

✅ **Pre-stimulus**: AUC ≈ 0.5 (chance level)
✅ **Early response** (100-200ms): AUC starts rising (early sensory processing)
✅ **Peak** (200-400ms): AUC peaks around 0.6-0.7 (oddball effect)
✅ **Late response** (400-500ms): AUC remains elevated or returns to baseline

**Group consistency**: 
- Individual subjects may vary in peak timing (±50ms)
- Grand average should show clear, sustained above-chance decoding
- Peak AUC typically 0.60-0.70 for well-preprocessed data

---

## Citation

If you use these scripts, consider citing:

**MNE-Python**:
```
Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python.
Frontiers in Neuroscience, 7, 267.
```

**Temporal Decoding Method**:
```
King & Dehaene (2014). Characterizing the dynamics of mental representations:
the temporal generalization method. Trends in Cognitive Sciences, 18(4), 203-210.
```

**Scikit-learn** (for classifier):
```
Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
```

---

## Questions?

Common questions answered:

**Q: Why use AUC instead of accuracy?**
A: AUC is more robust to class imbalance and provides better interpretation (0.5 = chance, 1.0 = perfect).

**Q: Should I use baseline correction?**
A: No - the StandardScaler handles normalization, and we want the classifier to learn from all available patterns.

**Q: Why equal number of epochs?**
A: Prevents the classifier from learning class frequency rather than actual patterns.

**Q: Can I decode more than 2 conditions?**
A: Yes - change to multiclass classification (use accuracy metric instead of AUC, and adapt the classifier).

**Q: What if my ISI is different?**
A: Adjust `tmax` to be less than your ISI to avoid contamination from the next trial.

**Q: How many trials do I need?**
A: Minimum 20 per condition, but 50+ is better for stable estimates.

---

## Files Summary

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `300_temporal_decoding_freq_vs_infreq.py` | Single subject analysis | Preprocessed .fif, bad channels, event dict | Plots + .npy results |
| `301_group_temporal_decoding.py` | Group analysis | All subjects' preprocessed data | Group plots + .npy + .csv |

Both scripts are self-contained and can be run independently. Start with `300_*` to test on one subject, then run `301_*` for the full group analysis.
