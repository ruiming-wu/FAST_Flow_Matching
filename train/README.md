# 🚂 train Module

This folder contains all scripts and logs related to training the neural network controllers for the Inverted Pendulum project, including both the original pi0 and the FAST (pi0-FAST) models.

---

## Overview

The `train` module provides end-to-end training pipelines for both continuous-action (pi0) and tokenized-action (pi0-FAST) transformer models. It supports dataset loading, model initialization, training, validation, testing, logging, and loss curve visualization.

---

## Main Files & Their Functions

- **train_pi0.py**  
  - Purpose: Trains the original pi0 model, which predicts a sequence of continuous actions using a flow-matching loss.
  - Features:
    - Loads training pairs from `data/training_pairs_original.npz`.
    - Uses a transformer model for flow matching vector field prediction.
    - Implements early stopping and learning rate scheduling.
    - Logs training/validation/test loss and saves the best model and loss curves.

- **train_pi0_fast.py**  
  - Purpose: Trains the pi0-FAST model, which predicts a sequence of discrete tokens representing compressed/quantized actions.
  - Features:
    - Loads training pairs from `data/training_pairs_fast.npz`.
    - Uses a transformer model for autoregressive token prediction.
    - Implements padding mask, early stopping, and learning rate scheduling.
    - Logs training/validation/test loss and saves the best model and loss curves.

---

## Training Details

- Both scripts save logs to `train/logs/`, models to `train/trained_models/`, and loss curves to `train/loss_pics/`.
- All training runs are reproducible and use the same data splits (80% train, 10% val, 10% test).
- Hyperparameters (such as learning rate, batch size, and embedding dimension) can be adjusted in each script.

---

## Model Ensemble for Evaluation

- For both **tiny pi0** and **tiny pi0fast**, a total of **10 models** were trained independently (with different random seeds).
- This enables robust evaluation: when comparing controller performance, the average and standard deviation across all 10 models can be reported for each method.

---

## Directory and File Naming Conventions

- **Logs (`logs/`)**  
  Each training run creates a log file named as:  
  `tinypi0_YYYYMMDD_HHMM.txt` or `tinypi0fast_YYYYMMDD_HHMM.txt`  
  where the prefix indicates the model type and the timestamp ensures uniqueness.

- **Trained Models (`trained_models/`)**  
  Each saved model checkpoint uses the same naming convention:  
  `tinypi0_YYYYMMDD_HHMM.pth` or `tinypi0fast_YYYYMMDD_HHMM.pth`  
  This allows you to easily identify and manage multiple runs and ensembles.

- **Loss Curves (`loss_pics/`)**  
  Each training run saves a loss curve plot as:  
  `tinypi0_YYYYMMDD_HHMM.png` or `tinypi0fast_YYYYMMDD_HHMM.png`  
  The file name matches the corresponding log and model checkpoint for easy reference.

---

## Example Usage

```bash
# Train a pi0 model
python -m train.train_pi0

# Train a pi0-FAST model
python -m train.train_pi0_fast
```