# RGMF: Reliable Guideline-Driven Multi-modal Fusion Model for Healthcare Analytics*

---

## Overview

Multimodal Electronic Health Records (EHRs), combining multivariate physiological time series and free-text clinical notes, offer rich signals for clinical outcome prediction. However, existing deep learning methods predominantly capture surface statistical correlations without grounding representations in evidence-based clinical standards, which limits their reliability in safety-critical settings. These black-box models also lack interpretability, making it difficult for clinicians to trace decision logic.

To address these issues, we propose **RGMF (Reliable Guideline-guided Multimodal Fusion)**, a framework that leverages clinical practice guidelines as knowledge anchors throughout the representation learning pipeline. RGMF first jointly encodes time-series vitals and clinical notes with timestamp-level embeddings on a unified timeline. A guideline-anchored prototype alignment module then initializes and calibrates modality-specific prototypes against the clinical guideline standard, ensuring representations conform to established medical criteria and resist noise-induced drift. An evidence-specificity dual-stream decision module integrates guideline-conformant evidence with patient-specific signals for balanced predictions.

Experiments on the **MIMIC-III** dataset demonstrate the effectiveness of RGMF in multi-modal EHR analysis.

---

## Architecture

RGMF consists of three core components:

1. **Unified Temporal Encoding**: Jointly encodes time-series vitals and clinical notes with timestamp-level embeddings on a unified timeline via multiTimeAttention (mTAND).
2. **Guideline-Anchored Prototype Alignment**: Initializes and calibrates modality-specific prototypes against clinical guideline standards using Slot Attention, ensuring representations conform to established medical criteria.
3. **Evidence-Specificity Dual-Stream Decision**: Integrates guideline-conformant evidence with patient-specific signals through a dual-branch decoder (guide branch + patient branch) for balanced, interpretable predictions.

For a detailed architecture diagram, see [`docs/framework.png`](docs/framework.png).
---

## Installation

### Requirements

- Python >= 3.8
- CUDA >= 11.7 (GPU recommended)

### Steps

```bash
# 1. Install system dependencies
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# 2. Install PyTorch (CUDA 12.1)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

---

## Dataset Preprocessing

RGMF is evaluated on the **MIMIC-III** dataset.

> **⚠️ Note:** You must obtain access to MIMIC-III by applying at [PhysioNet](https://physionet.org/content/mimiciii/1.4/).

### Step-by-Step Preprocessing

1. Download the original [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) database to your disk.

2. Run the preprocessing pipeline:

```bash
cd src/cmehr/preprocess
```

3. Extract subject directories:

```bash
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

4. Validate and clean events (~80% of events remain):

```bash
python -m mimic3benchmark.scripts.validate_events data/root/
```

5. Break data into ICU stay episodes:

```bash
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

6. Split into training and testing sets:

```bash
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

7. Generate task-specific datasets:

```bash
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
```

8. Extract validation set:

```bash
python -m mimic3models.split_train_val {dataset-directory} --valset mimic3models/resources/valset_{task}.csv
```

9. Generate clinical note features:

```bash
cd src/cmehr/preprocess/ClinicalNotesICU/mimic3
# Modify dataset_path and output_folder in extract_notes.py, then run:
python extract_notes.py
python extract_T0.py
```

10. Create multimodal pickle files:

```bash
python -m mimic3models.create_iiregular_ts --task {TASK}
```

> **Note:** Update the pickle save path in `src/cmehr/paths.py` (Line 8) before running.

---

## Usage

Before running, configure the following in `src/cmehr/paths.py`:

```python
DATA_PATH = Path("/path/to/your/output_mimic3")
```

### Run RGMF

```bash
cd scripts/mimic3
sh train_rgmf_ihm.sh    # In-hospital mortality prediction
sh train_rgmf_pheno.sh   # 24-hour phenotype classification
```

### Run Baselines

```bash
sh ts_baselines.sh       # Time-series baseline models
sh note_baselines.sh     # Clinical note baseline models
```

> **Note:** Baseline model implementations are sourced from their official repositories.

---

## Project Structure

```
CTPD/                          # Root directory
├── src/cmehr/                 # Source code
│   ├── backbone/              # Backbone encoders (time-series & vision)
│   │   ├── time_series/       # FCN, ResNet, InceptionTime
│   │   └── vision/            # Vision transformer, pretrained encoders
│   ├── dataset/               # Data modules for MIMIC-III and MIMIC-IV
│   │   ├── mimic3_downstream_datamodule.py
│   │   └── mimic4_downstream_datamodule.py
│   ├── models/                # Model implementations
│   │   ├── mimic3/           # MIMIC-III models (incl. RGMF)
│   │   │   └── rgmf_model.py
│   │   └── mimic4/          # MIMIC-IV models (incl. RGMF)
│   │       └── rgmf_model.py
│   ├── preprocess/            # Data preprocessing scripts
│   └── utils/                # Utilities (losses, schedulers, metrics)
├── scripts/                   # Training & evaluation scripts
│   ├── mimic3/
│   │   ├── train_mimic3.py   # Main training script
│   │   ├── train_rgmf_ihm.sh
│   │   └── train_rgmf_pheno.sh
│   └── mimic4/
├── docs/                      # Documentation
│   ├── framework.png
│   └── RGMF-TGCL-Framework.md
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Acknowledgements

We thank the awesome open-source repositories that this project builds upon:

- [MIMIC-III Benchmark](https://github.com/YerevaNN/mimic3-benchmarks)
- [ClinicalNotesICU](https://github.com/kaggarwal/ClinicalNotesICU)
- [MultimodalMIMIC](https://github.com/XZhang97666/MultimodalMIMIC)
- [MIMIC-IV Benchmark](https://github.com/mimic-iv/mimic-iv)
- [BioViL-T](https://github.com/microsoft/hi-ml)

---
