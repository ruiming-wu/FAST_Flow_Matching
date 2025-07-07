# 🔎 inference Module

This folder contains all scripts for evaluating and visualizing the performance of trained controllers (pi0 and pi0-FAST) in the Inverted Pendulum environment. It supports both single-model and ensemble evaluation, as well as interactive environment rollouts and batch statistical analysis.

---

## Overview

The `inference` module provides tools for:
- Running trained models in the MuJoCo Inverted Pendulum environment with visualization.
- Batch evaluation of multiple models to compute average and standard deviation of performance metrics.
- Decoding and post-processing of model outputs (including token-to-action conversion for pi0-FAST).
- Flexible configuration for random initial states, noise injection, and replanning intervals.

---

## Main Files & Their Functions

- **inference_pi0.py**  
  - Purpose: Provides the function to generate a continuous action sequence from a trained pi0 model using flow matching and Euler integration.
  - Example:  
    ```python
    from inference.inference_pi0 import infer_pi0_action_sequence
    actions = infer_pi0_action_sequence(model_path, state_vec)
    ```

- **inference_pi0fast.py**  
  - Purpose: Provides the function to generate a token sequence from a trained pi0-FAST model, using autoregressive decoding.
  - Example:  
    ```python
    from inference.inference_pi0fast import infer_pi0fast_token_sequence
    tokens = infer_pi0fast_token_sequence(model_path, state_vec)
    ```

- **env_pi0.py**  
  - Purpose: Runs a trained pi0 model in the environment, visualizes the trajectory, and supports interactive rendering and random initial states.
  - Usage:  
    ```bash
    python -m inference.env_pi0
    ```

- **env_pi0fast.py**  
  - Purpose: Runs a trained pi0-FAST model in the environment, including token decoding, inverse DCT, and visualization.
  - Usage:  
    ```bash
    python -m inference.env_pi0fast
    ```

- **main_pi0.py**  
  - Purpose: Batch evaluation of all pi0 models in a directory, reporting success rate, convergence statistics, and per-model metrics over many random runs.
  - Output: Prints summary statistics and returns all raw data for further analysis.

- **main_pi0fast.py**  
  - Purpose: Batch evaluation of all pi0-FAST models in a directory, including token decoding and action reconstruction, with the same statistical reporting as main_pi0.py.

---

## Typical Workflow

1. **Interactive Rollout & Visualization**  
   Use `env_pi0.py` or `env_pi0fast.py` to run a single model in the environment and visualize the trajectory and actions.

2. **Batch Evaluation for Statistics**  
   Use `main_pi0.py` or `main_pi0fast.py` to evaluate all models (e.g., 10-model ensemble) and compute average performance, convergence, and robustness metrics.

3. **Custom Inference**  
   Use `inference_pi0.py` or `inference_pi0fast.py` as utility functions in your own scripts for custom evaluation or integration.

---

## Notes

- All scripts are fully compatible with both CPU and GPU.
- The pi0-FAST pipeline includes token decoding and inverse DCT to reconstruct continuous actions from discrete tokens.
- Batch evaluation scripts (`main_pi0.py`, `main_pi0fast.py`) are designed for reproducible, large-scale statistical analysis and support random initial state, noise injection and different replan interval.
- The module assumes that trained models and tokenizers are saved in the standard locations as produced by the `train` and `fast` modules.

---