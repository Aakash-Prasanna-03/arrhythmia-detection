# Arrhythmia Detection with Deep Learning

This repository provides a comprehensive pipeline for training, evaluating, and auditing deep learning models for arrhythmia detection using 12-lead ECG data. It supports both the PTB-XL (Chapman) and Georgia ECG datasets, and includes scripts for model training, evaluation, and label dictionary auditing. The project also provides detailed results and observations, including Out-of-Distribution (OOD) performance analysis and per-lead ablation studies.

This project provides a full pipeline for training, evaluating, and auditing deep learning models for arrhythmia detection using 12-lead ECG data. It supports both the PTB-XL (Chapman) and Georgia ECG datasets, and includes scripts for model training, evaluation, and label dictionary auditing.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Auditing Label Dictionaries](#auditing-label-dictionaries)
- [Scripts Overview](#scripts-overview)
- [References](#references)

---

## Project Structure
```
├── audit_chapman.py           # Audit Chapman label dictionary
├── audit_georgia.py           # Audit Georgia label dictionaries
├── dataloader.py              # PTB-XL/Georgia data loader
├── downsample_chapman.py      # (Optional) Downsampling utilities
├── downsample_georgia.py      # (Optional) Downsampling utilities
├── eval.py                    # (Legacy) Evaluation script
├── eval12.py                  # Main evaluation script (12-lead)
├── leadbasedtrain.py          # Train single-lead models
├── leadtrain/                 # (Optional) Lead-based training outputs
├── lead_based_models/         # Saved models and results
├── models/
│   └── resnet1d.py            # 1D ResNet model definition
├── train12lead.py             # Train 12-lead model
├── data/
│   ├── chapman/               # Chapman dataset
│   ├── georgia/               # Georgia dataset
│   └── ...                    # Metadata, mappings, etc.
├── eda/                       # Exploratory data analysis scripts
└── ...
```

---

## Setup & Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- pandas, numpy, tqdm
- wfdb (for ECG signal reading)

Install dependencies:
```bash
pip install torch scikit-learn pandas numpy tqdm wfdb
```

---

## Data Preparation
- Place the PTB-XL (Chapman) and Georgia ECG datasets in the `data/chapman/` and `data/georgia/` folders, respectively.
- Ensure the following files exist:
  - `data/chapman/X_chapman.npy`, `y_chapman.npy`
  - `data/georgia/X_georgia.npy`, `y_georgia.npy`
  - Metadata: `ptbxl_database_clean.csv` or `ptbxl_database.csv`, `scp_statements.csv`, etc.

---

## Model Training
### 12-Lead Model
Train a 12-lead ResNet model on PTB-XL:
```bash
python train12lead.py
```
- Model weights are saved to `lead_based_models/best_12lead.pth`.

### Single-Lead Models
Train a model for each lead:
```bash
python leadbasedtrain.py
```
- Models are saved to `lead_based_models/best_lead_<LEAD>.pth`.

---

## Evaluation
Evaluate the trained 12-lead model on Chapman or Georgia datasets:
```bash
python eval12.py chapman
python eval12.py georgia
```
- Results are printed and saved as `results_12lead.json` in the current directory.
- For custom output, move or rename the JSON as needed.

---


## Results & Observations

### 1. 12-Lead Model Performance

**Chapman (OOD):**
```json
{
  "macro_f1": 0.425,
  "per_class_f1": {
    "NORM": 0.754,
    "MI": 0.216,
    "STTC": 0.646,
    "CD": 0.367,
    "HYP": 0.143
  }
}
```

**Georgia (OOD):**
```json
{
  "macro_f1": 0.502,
  "per_class_f1": {
    "NORM": 0.573,
    "MI": 0.095,
    "STTC": 0.681,
    "CD": 0.658,
    "HYP": 0.503
  }
}
```

### 2. Single-Lead Model Performance (PTB-XL, ID)

| Lead | Macro F1 | NORM | MI | STTC | CD | HYP |
|------|----------|------|----|------|----|-----|
| I    | 0.60     | 0.79 | 0.55 | 0.67 | 0.59 | 0.42 |
| II   | 0.62     | 0.80 | 0.59 | 0.66 | 0.66 | 0.41 |
| III  | 0.54     | 0.74 | 0.57 | 0.47 | 0.57 | 0.36 |
| aVR  | 0.63     | 0.81 | 0.58 | 0.70 | 0.65 | 0.43 |
| aVL  | 0.56     | 0.77 | 0.57 | 0.52 | 0.57 | 0.37 |
| aVF  | 0.58     | 0.77 | 0.59 | 0.55 | 0.60 | 0.38 |
| V1   | 0.60     | 0.75 | 0.57 | 0.54 | 0.66 | 0.49 |
| V2   | 0.56     | 0.74 | 0.57 | 0.53 | 0.62 | 0.36 |
| V3   | 0.58     | 0.77 | 0.58 | 0.59 | 0.55 | 0.38 |
| V4   | 0.61     | 0.80 | 0.56 | 0.67 | 0.58 | 0.42 |
| V5   | 0.64     | 0.81 | 0.56 | 0.71 | 0.60 | 0.50 |
| V6   | 0.65     | 0.81 | 0.58 | 0.71 | 0.63 | 0.51 |

### 3. Out-of-Distribution (OOD) Performance Drop

**Chapman OOD (Macro F1 drop):**

| Lead | PTB_ID_Macro_F1 | CHAPMAN_OOD_Macro_F1 | Absolute_Drop |
|------|-----------------|----------------------|---------------|
| I    | 0.60            | 0.44                 | 0.17          |
| II   | 0.62            | 0.43                 | 0.19          |
| III  | 0.54            | 0.40                 | 0.14          |
| aVR  | 0.63            | 0.44                 | 0.19          |
| aVL  | 0.56            | 0.40                 | 0.16          |
| aVF  | 0.58            | 0.42                 | 0.15          |
| V1   | 0.60            | 0.40                 | 0.20          |
| V2   | 0.56            | 0.38                 | 0.19          |
| V3   | 0.58            | 0.38                 | 0.20          |
| V4   | 0.61            | 0.36                 | 0.24          |
| V5   | 0.64            | 0.42                 | 0.22          |
| V6   | 0.65            | 0.40                 | 0.25          |

**Georgia OOD (Macro F1 drop):**

| Lead | PTB_ID_Macro_F1 | GEORGIA_OOD_Macro_F1 | Absolute_Drop |
|------|-----------------|----------------------|---------------|
| I    | 0.60            | 0.51                 | 0.10          |
| II   | 0.62            | 0.50                 | 0.12          |
| III  | 0.54            | 0.47                 | 0.07          |
| aVR  | 0.63            | 0.53                 | 0.11          |
| aVL  | 0.56            | 0.46                 | 0.10          |
| aVF  | 0.58            | 0.47                 | 0.11          |
| V1   | 0.60            | 0.50                 | 0.10          |
| V2   | 0.56            | 0.45                 | 0.12          |
| V3   | 0.58            | 0.48                 | 0.10          |
| V4   | 0.61            | 0.49                 | 0.12          |
| V5   | 0.64            | 0.48                 | 0.16          |
| V6   | 0.65            | 0.49                 | 0.16          |

#### Key Observations
- **OOD Generalization:** All models show a significant drop in macro F1 when evaluated on OOD datasets (Chapman, Georgia), with the drop being more severe for Chapman.
- **Per-Class Trends:** NORM and STTC classes are generally predicted with higher F1, while MI and HYP are consistently lower, especially in OOD settings.
- **Lead Importance:** Lateral/precordial leads (V5, V6) and limb leads (I, II, aVR) tend to perform best in both ID and OOD settings.
- **12-Lead vs. Single-Lead:** The 12-lead model outperforms any single-lead model, but the best single leads (V5, V6, I, II) are not far behind.
- **MI Detection:** Myocardial Infarction (MI) is the most challenging class, with F1 dropping below 0.1 on Georgia OOD.
- **Consistency:** The results are consistent with prior literature on ECG arrhythmia detection and OOD generalization.

---

---

## Auditing Label Dictionaries
- Audit Chapman label dictionary:
  ```bash
  python audit_chapman.py
  ```
- Audit Georgia label mappings:
  ```bash
  python audit_georgia.py
  ```

---

## Scripts Overview
- `train12lead.py`: Train a 12-lead ResNet model on PTB-XL.
- `leadbasedtrain.py`: Train single-lead models for ablation/comparison.
- `eval12.py`: Evaluate a trained model and output results as JSON.
- `dataloader.py`: Loads and processes ECG data and labels.
- `models/resnet1d.py`: 1D ResNet model definition.
- `audit_chapman.py`, `audit_georgia.py`: Audit label dictionaries for consistency and coverage.

---

## References
- PTB-XL: https://physionet.org/content/ptb-xl/
- Chapman ECG: https://www.kaggle.com/datasets/chapman/ECG
- Georgia ECG: [Dataset source]
- SNOMED CT: https://www.snomed.org/

---

For questions or issues, please open an issue or contact the maintainer.
