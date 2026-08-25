# RSNA Knee Abnormality Detection

A single knee scan can reveal a dozen different problems. This repo holds my work for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: building a model that detects a defined set of clinically important
abnormalities on knee MRI examinations.

**Task:** predict per-study probabilities for 12 knee MRI findings.
**Metric:** macro-averaged AUC ROC across the 12 labels.
**Final submission deadline:** October 22, 2026.
**Entry / team merger deadline:** October 15, 2026.

## Layout

```
configs/      experiment configs (one YAML per run)
docs/         plans, notes, learning material
notebooks/    exploration and Kaggle submission notebooks
src/          reusable pipeline code
  data/       DICOM loading, series selection, dataset classes
  labels/     deriving labels from radiology reports
  models/     image and text model definitions
  train/      training loops, CV, checkpointing
  eval/       metric, OOF analysis
  submit/     submission assembly
```

## The 12 targets

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

## Setup

```bash
pip install -r requirements.txt
```

Then place or symlink the Kaggle data at `data/raw/`.

## Status

Scaffold stage. No trained model yet. 
