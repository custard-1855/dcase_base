# DCASE SED Codebase - Documentation Index

This repository contains a comprehensive Sound Event Detection (SED) system for DCASE 2024 Task 4 with detailed documentation. Use this index to navigate the codebase analysis.

## Documentation Files

### 1. **DCASE_ANALYSIS.md** (21 KB) - Main Technical Reference
Complete technical documentation of the codebase with 10 major sections:

- **Section 1:** SED Models Implemented
  - CRNN (Convolutional Recurrent Neural Network) - main architecture
  - BEATs - pre-trained audio embeddings
  - CNN and RNN components
  - Configuration parameters

- **Section 2:** Inference/Prediction Code
  - Training module methods (detect, validation_step, test_step)
  - Embedding extraction pipeline
  - Post-processing (median filtering, cSEBBs)

- **Section 3:** Audio Data Processing
  - Audio reading and preprocessing functions
  - Feature extraction (mel-spectrograms)
  - Data augmentation techniques
  - Dataset classes

- **Section 4:** Dependencies and Libraries
  - Audio processing (torchaudio, librosa, scipy)
  - Deep learning frameworks (torch, pytorch-lightning)
  - Evaluation metrics (sed-scores-eval, psds-eval)

- **Section 5:** Real-time/Streaming Inference
  - Current state (no dedicated real-time module)
  - Inference bottlenecks
  - Potential streaming implementation

- **Section 6:** Project Structure
  - Directory tree with file descriptions
  - Module organization

- **Sections 7-10:** Training, Configuration, Experiment Tracking, Summary

**Use this for:** Complete technical understanding, implementation details, parameter explanations.

---

### 2. **QUICK_REFERENCE.md** (7.8 KB) - Developer Quick Reference
Fast-lookup guide for developers:

- **Critical File Paths:** All absolute paths to key files
- **Code Snippets:** 6 practical examples
  1. Load pre-trained model
  2. Perform inference on audio
  3. Extract BEATs embeddings
  4. Create custom model
  5. Train custom model
  6. Load data

- **Key Parameters Summary:** Network architecture, training, features
- **Common Issues & Solutions:** OOM, missing embeddings, mismatched labels, slow training
- **Evaluation Metrics:** PSDS, F1 scores, intersection-based metrics
- **File Organization:** Quick view of structure

**Use this for:** Quick lookups, copy-paste code, debugging common issues.

---

### 3. **ARCHITECTURE.md** (9.3 KB) - System Architecture Diagram
Visual and text-based architecture overview:

- **System Overview:** ASCII diagram showing entire pipeline
- **8 Major Stages:**
  1. Input (audio files, preprocessing)
  2. Embedding (BEATs model)
  3. Main Model (CRNN: CNN + RNN + embedding fusion)
  4. Training (Mean Teacher, semi-supervised learning)
  5. Inference (prediction)
  6. Post-processing (median filtering, cSEBBs)
  7. Evaluation (metrics computation)
  8. Data Flow (end-to-end flow)

- **Key Dimensions:** Tensor shapes throughout the pipeline
- **Layer Details:** Exact filter sizes, pooling, output dimensions

**Use this for:** Understanding system flow, debugging tensor shape issues, architecture visualization.

---

## Quick Navigation

### By Use Case

**I want to...**

- **Understand the model architecture** → Start with `ARCHITECTURE.md` section 3 + `DCASE_ANALYSIS.md` section 1
- **Write inference code** → See `QUICK_REFERENCE.md` snippets 1-2 + `DCASE_ANALYSIS.md` section 2
- **Load a dataset** → Use `QUICK_REFERENCE.md` snippet 6 + `DCASE_ANALYSIS.md` section 3
- **Extract embeddings** → Check `QUICK_REFERENCE.md` snippet 3 + `DCASE_ANALYSIS.md` section 3.5
- **Train a model** → See `QUICK_REFERENCE.md` snippet 5 + `DCASE_ANALYSIS.md` section 7
- **Debug a problem** → Check `QUICK_REFERENCE.md` "Common Issues & Solutions"
- **Find a specific file** → Use `QUICK_REFERENCE.md` "Critical File Paths"
- **Understand the full system** → Read all of `ARCHITECTURE.md` then `DCASE_ANALYSIS.md`

---

### By Document Type

| Need | Document | Section |
|------|----------|---------|
| **File locations** | QUICK_REFERENCE | Critical File Paths |
| **Code examples** | QUICK_REFERENCE | Code Snippets |
| **Detailed implementation** | DCASE_ANALYSIS | All sections 1-10 |
| **System overview** | ARCHITECTURE | All sections |
| **Tensor shapes** | ARCHITECTURE | Key Dimensions |
| **Troubleshooting** | QUICK_REFERENCE | Common Issues |
| **Data formats** | DCASE_ANALYSIS | Section 3 |
| **Parameters** | QUICK_REFERENCE | Key Parameters Summary |

---

## Document Statistics

```
Total Documentation: 38 KB
├─ DCASE_ANALYSIS.md: 21 KB (Comprehensive technical reference)
├─ QUICK_REFERENCE.md: 7.8 KB (Developer quick reference)
├─ ARCHITECTURE.md: 9.3 KB (System architecture)
└─ DOCUMENTATION_INDEX.md: 0.9 KB (This file)

Code Coverage:
├─ Model architectures: 4 files analyzed (CRNN, CNN, RNN, BEATs)
├─ Training pipeline: 2 files (train_pretrained.py, sed_trainer_pretrained.py)
├─ Inference: 3 files (extract_embeddings.py, test_step, postprocess)
├─ Data processing: 5 files (datasets, pre_datasets, data_augm, encoder, scaler)
├─ Configuration: 3 YAML files documented
└─ Total: 47 Python files analyzed
```

---

## Key Facts About the Codebase

1. **Framework:** PyTorch + PyTorch Lightning
2. **Main Task:** Weakly-supervised Sound Event Detection with domain adaptation
3. **Datasets:** DESED (Domestic Sound Event Detection) + MAESTRO (musical instruments)
4. **Model Type:** CRNN (Convolutional Recurrent Neural Network)
5. **Pre-trained:** BEATs (Audio Pre-Training with Acoustic Tokenizers)
6. **Learning:** Semi-supervised Mean Teacher framework
7. **Languages:** 27 sound event classes
8. **Audio:** 16 kHz, 10-second clips, mono
9. **Features:** 128 mel-bins spectrograms
10. **Post-processing:** cSEBBs (change-point detection) + median filtering

---

## File Access Guide

### To Access Critical Files
All files use absolute paths like:
```
/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/...
```

See `QUICK_REFERENCE.md` "Critical File Paths" section for complete list.

### Main Entry Points
- **Training:** `train_pretrained.py`
- **Inference:** `extract_embeddings.py` or `test_step` in `sed_trainer_pretrained.py`
- **Configuration:** `confs/pretrained.yaml`

---

## How to Use These Documents

### For First-Time Users
1. Read: `ARCHITECTURE.md` (5 min) - understand the system
2. Read: `DCASE_ANALYSIS.md` sections 1-3 (10 min) - learn key components
3. Reference: `QUICK_REFERENCE.md` as needed

### For Implementation
1. Find file path: `QUICK_REFERENCE.md` → "Critical File Paths"
2. Copy code: `QUICK_REFERENCE.md` → "Code Snippets"
3. Understand details: `DCASE_ANALYSIS.md` → relevant section
4. Visualize: `ARCHITECTURE.md` → system diagram

### For Debugging
1. Check issue: `QUICK_REFERENCE.md` → "Common Issues & Solutions"
2. Verify tensor shapes: `ARCHITECTURE.md` → "Key Dimensions"
3. Read implementation: `DCASE_ANALYSIS.md` → relevant section

---

## Related Code Files

### Core Model
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/CRNN.py`
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/beats/BEATs.py`

### Training
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/train_pretrained.py`
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py`

### Data
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/dataio/datasets.py`
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/data_augm.py`

### Inference
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/extract_embeddings.py`
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/sebbs/sebbs/csebbs.py`

---

## Notes

- All file paths are **absolute paths** (not relative)
- Code snippets are **production-ready** but may need adaptation
- Configuration uses **YAML format** for easy modification
- Dependency management uses **uv** (not pip)
- Experiment tracking via **Weights & Biases (wandb)**
- Framework: **PyTorch Lightning 1.9.x**

---

## Version Info

- **Project:** DCASE 2024 Task 4 Baseline
- **Documentation created:** 2025-11-02
- **Codebase branch:** dev
- **Last commit:** add save ckpt to wandb

---

**For the most detailed information, see DCASE_ANALYSIS.md**

**For quick lookups and examples, see QUICK_REFERENCE.md**

**For system visualization, see ARCHITECTURE.md**
