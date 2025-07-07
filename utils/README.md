# 🛠️ utils Module

This folder contains utility scripts for data processing, batch checking, and preparation of training data for the Inverted Pendulum project. These scripts are essential for converting raw trajectory data into formats suitable for model training and evaluation.

---

## Overview

The `utils` module provides scripts for:
- Batch collection and checking of trajectory data.
- Building training pairs for both the original and FAST models.
- Constructing the corpus and vocabulary for the custom tokenizer.

---

## Main Files & Their Functions

- **batch_collection.py**  
  - Purpose: Automates the batch collection of trajectories by repeatedly calling the data collection script for different indices.
  - Usage:  
    ```bash
    python -m utils.batch_collection
    ```

- **batch_check.py**  
  - Purpose: Checks all trajectory files for correct shape and convergence, and reports statistics such as the number of valid files and average convergence step.
  - Output: Prints summary statistics to the console.

- **build_pairs_original.py**  
  - Purpose: Processes all trajectory files to generate training pairs for the original pi0 model. Each pair consists of an initial state and a sequence of 50 future actions.
  - Output: Saves `data/training_pairs_original.npz`.

- **build_pairs_fast.py**  
  - Purpose: Encodes the quantized action corpus into token ID sequences using the trained tokenizer, and aligns them with the corresponding states to create training pairs for the pi0-FAST model.
  - Output: Saves `data/training_pairs_fast.npz`.

- **build_corpus.py**  
  - Purpose: Converts action sequences into quantized DCT coefficients, builds the corpus for tokenizer training, and generates the base vocabulary file.
  - Output: Saves `fast/tokenizer/fast_tokenizer_corpus.txt` and `fast/tokenizer/fast_tokenizer_base_vocab.txt`.

---

## Typical Workflow

1. **Batch Collect Trajectories**  
   Use `batch_collection.py` to generate a large number of trajectory files.

2. **Check Data Quality**  
   Use `batch_check.py` to verify the integrity and convergence of all collected trajectories.

3. **Build Training Pairs for pi0**  
   Use `build_pairs_original.py` to create state-action training pairs for the original model.

4. **Build Corpus and Vocabulary for Tokenizer**  
   Use `build_corpus.py` to generate the corpus and base vocabulary for the custom tokenizer.

5. **Build Training Pairs for pi0-FAST**  
   Use `build_pairs_fast.py` to encode the corpus and create tokenized training pairs for the FAST model.

---

## Notes

- All scripts are designed to be run from the project root directory.
- The outputs of these scripts are used by the `train` and `fast` modules for model training and tokenizer construction.
- Ensure that the required input files (such as trajectory `.npy` files) exist before running the corresponding utility scripts.