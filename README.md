# Leakage-Free Optical Network Fault Detection with Machine Learning

A reproducible Python implementation for **lightpath-level, leakage-free soft-failure detection** in optical-network telemetry. The repository also includes an optional regression path for **gain estimation** when a numeric gain target is available.

The code is designed to prevent a common evaluation error in telemetry research: allowing measurements from the same lightpath to appear in both training and test sets.

## Main Features

- Lightpath/group-level train-test separation
- Automatic leakage check before evaluation
- Autoencoder trained on **No-Failure / normal** training telemetry only
- Threshold calibration using **training validation data only**
- ROC-AUC, F1, precision, recall, and accuracy reporting
- Optional raw-feature vs. physics-inspired feature ablation
- Optional supervised gain-regression module
- Flexible CSV/TXT delimiter detection
- No absolute paths or machine-specific configuration

## Reported Experimental Result

The associated project evaluated the full optical soft-failure benchmark at the lightpath level using a leakage-free split of **605 training lightpaths** and **151 held-out test lightpaths**.

| Metric | Reported result |
|---|---:|
| ROC-AUC | 0.9997 ± 0.0000 |
| F1 score | 0.9965 ± 0.0004 |
| Seeds | 5 |
| EDFA recall | 0.996 |

A raw-telemetry ablation reported an F1 score of approximately **0.9966**, indicating that the strong detection performance was not dependent on the added physics-engineered features.

> These numbers are included as the reported results of the original project. Re-running the code can produce different values depending on dataset version, column naming, split, preprocessing, and random seed.

## Dataset

The original archive contains very large telemetry files, including one file larger than GitHub's normal single-file upload limit. They are therefore intentionally **not bundled in this repository**.

Expected file names in the original project include:

```text
Lightpath_756_label_4_QoT_dataset_train_100.txt
Lightpath_756_label_4_QoT_dataset_train_900.txt
Lightpath_756_label_4_QoT_dataset_test_300.txt
```

See `DATASET.md` for setup instructions.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1 — One dataset file

The program creates a leakage-free lightpath-level split internally:

```bash
python optical_fault_detection.py \
  --data Lightpath_756_label_4_QoT_dataset.txt \
  --group-column lightpath_id \
  --label-column failure_type \
  --normal-label No-Failure
```

### Option 2 — Predefined train and test files

```bash
python optical_fault_detection.py \
  --train Lightpath_756_label_4_QoT_dataset_train_900.txt \
  --test Lightpath_756_label_4_QoT_dataset_test_300.txt \
  --group-column lightpath_id \
  --label-column failure_type \
  --normal-label No-Failure
```

### Physics-feature ablation

```bash
python optical_fault_detection.py \
  --train TRAIN_FILE.txt \
  --test TEST_FILE.txt \
  --group-column lightpath_id \
  --label-column failure_type \
  --normal-label No-Failure \
  --physics-features
```

### Optional gain estimation

If the dataset contains a numeric target such as `gain_db`:

```bash
python optical_fault_detection.py \
  --train TRAIN_FILE.txt \
  --test TEST_FILE.txt \
  --group-column lightpath_id \
  --label-column failure_type \
  --normal-label No-Failure \
  --gain-target gain_db
```

## Evaluation Protocol

1. Split data at the **lightpath/group level**, never by individual telemetry row.
2. Fit preprocessing on training data only.
3. Use only normal training samples to train the autoencoder.
4. Calibrate the anomaly threshold on a validation subset derived from training lightpaths only.
5. Evaluate once on the held-out lightpaths.
6. Repeat with multiple random seeds when reporting final statistics.

This prevents lightpath leakage and avoids using test data for threshold selection.

## Output

The default output is `results.json`, containing:

- ROC-AUC
- F1 score
- precision
- recall
- accuracy
- calibrated threshold
- number of train/test rows
- number of train/test lightpaths
- feature count
- optional gain-regression metrics

## Important Notes

- Column names differ across dataset versions. Use `--group-column` and `--label-column` when automatic detection is not appropriate.
- Physics-inspired feature interactions in this repository are intentionally conservative and are provided for ablation. They should not be described as physical compensation.
- Real nonlinearity compensation requires appropriate waveform/IQ data and a compensation algorithm; anomaly detection alone is not compensation.
- Large raw datasets should normally be hosted through the original dataset provider, Zenodo, Figshare, Kaggle, or Git LFS rather than committed directly to a standard GitHub repository.

## Requirements

- Python 3.10+
- NumPy
- pandas
- scikit-learn

## License

The source code in this repository is released under the MIT License. Dataset files remain subject to the license and terms of their original provider.

## Citation

If you use this repository in academic work, please cite the associated publication/thesis when available and include the repository URL and commit used for your experiment.
