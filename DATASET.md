# Dataset Setup

The raw telemetry files are intentionally excluded from this GitHub-ready package because the original project contains files that are too large for a normal GitHub commit and may also be governed by the dataset provider's own redistribution terms.

## Original project file names

```text
Lightpath_756_label_4_QoT_dataset_train_100.txt
Lightpath_756_label_4_QoT_dataset_train_900.txt
Lightpath_756_label_4_QoT_dataset_test_300.txt
```

Place the required dataset files in the same directory as `optical_fault_detection.py`, or pass an absolute/relative path through `--data`, `--train`, and `--test`.

## GitHub size note

GitHub blocks normal commits of individual files above its standard file-size limit. Do not upload the very large raw telemetry files directly to the repository. Prefer one of the following:

- the original public dataset link,
- Zenodo / Figshare,
- Kaggle datasets,
- Git LFS when redistribution is permitted.

## Required columns

The script needs:

1. a lightpath/group identifier,
2. a failure/condition label,
3. one or more numeric telemetry features.

If automatic column detection does not match your dataset, explicitly provide:

```bash
--group-column YOUR_LIGHTPATH_COLUMN
--label-column YOUR_LABEL_COLUMN
--normal-label YOUR_NORMAL_CLASS
```

For optional gain estimation, also provide:

```bash
--gain-target YOUR_NUMERIC_GAIN_COLUMN
```
