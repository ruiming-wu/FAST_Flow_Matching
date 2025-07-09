# tum-adlr-ss25-12-FAST

A small-scale control modeling project comparing FAST with original action chunk / autoregressive generation with flow matching using Transformers in CartPole (Inverted Pendulum) simulation.

---

## 🧭 Project Overview

This project implements and compares two control pipelines for the Inverted Pendulum task:
- **pi0**: Transformer-based flow matching for direct continuous action sequence prediction.
- **pi0-FAST**: Transformer-based autoregressive token generation for compressed/quantized action sequences using a custom tokenizer (FAST method).

The workflow covers data collection, preprocessing, model training, benchmarking, and interactive or visual evaluation.

---

## 📂 Repository Structure

```
.
├── data/                # All datasets, processed files, and training pairs
├── data_collection/     # Scripts for collecting and visualizing trajectories (PID control)
├── fast/                # Custom tokenizer, encoder/decoder, and action compression for FAST method
├── inference/           # Scripts for running and visualizing model inference and evaluation
├── model/               # Transformer architectures for pi0 and pi0-FAST
├── train/               # Training scripts, logs, checkpoints, and loss curves
├── utils/               # Data processing, batch checking, benchmarking, and visualization utilities
├── documentations/      # Project diagrams, flowcharts, and supplementary materials
└── README.md            # Project introduction and usage guide
```

Each module contains a dedicated README with detailed explanations and usage instructions.

---

## 📝 Module Summaries

### data/
- Stores all raw and processed data, including trajectories, parameters, visualizations, and training pairs for both models.
- See [data/README.md](./data/README.md) for details.

### data_collection/
- Scripts for collecting trajectories using PID control, saving data, and visualizations.
- See [data_collection/README.md](./data_collection/README.md).

### fast/
- Implements the custom tokenizer and action compression for the FAST method.
- Includes encoder/decoder and tokenizer training scripts.
- See [fast/README.md](./fast/README.md).

### inference/
- Scripts for running trained models in the environment, visualizing rollouts, and batch evaluation.
- See [inference/README.md](./inference/README.md).

### model/
- PyTorch implementations of transformer architectures for both pi0 and pi0-FAST.
- See [model/README.md](./model/README.md).

### train/
- Training pipelines, logs, checkpoints, and loss curves for both models.
- See [train/README.md](./train/README.md).

### utils/
- Utility scripts for data processing, batch checking, benchmarking, and visualization.
- See [utils/README.md](./utils/README.md).

---

## 💡 Tips for New Users

- **Read each module's README:**  
  Each folder contains a detailed README with usage instructions and workflow.
- **Start with the data pipeline:**  
  Use the provided scripts to collect, check, and process data before training.
- **Try interactive demos:**  
  Use the inference scripts for interactive or visual exploration of model performance.
- **For research or extension:**  
  The modular design allows easy adaptation to other control tasks or model architectures.

---

## 🚀 Quick Start

### 1. Data Collection

- Collect single or batch trajectories using PID controllers:
  ```bash
  python -m data_collection.main_pid --idx 4001
  # Or batch collection (please change the range of idx manually):
  python -m utils.batch_collection
  ```
- Trajectories, parameters, and visualizations are saved in `data/`.

### 2. Data Processing

- Check data quality and build training pairs:
  ```bash
  python -m utils.batch_check
  python -m utils.build_pairs_original
  python -m utils.build_corpus
  python -m fast.train_tokenizer
  python -m utils.build_pairs_fast
  ```

### 3. Model Training

- Train the original pi0 model:
  ```bash
  python -m train.train_pi0
  ```
- Train the pi0-FAST model:
  ```bash
  python -m train.train_pi0_fast
  ```

### 4. Inference, Visualization & Evaluation

- Visualize and compare model predictions:
  ```bash
  python -m utils.show_inference
  ```
- Benchmark inference speed:
  ```bash
  python -m utils.time_pi0
  python -m utils.time_pi0fast
  ```
- Interactive rollout in environment:
  ```bash
  python -m inference.env_pi0
  python -m inference.env_pi0fast
  ```
- Batch evaluation for statistics:
  ```bash
  python -m inference.main_pi0
  python -m inference.main_pi0fast
  ```

---

## 🖼️ Visualization & Interactive Demo

- **Trajectory Visualization:**  
  All collected and generated trajectories (you need to specify a chunk idx for the model to reason about) can be visualized as plots (see `data/trajs_pics/` and `utils/show_inference.py`).
- **Interactive Environment:**  
  Run `inference/env_pi0.py` or `inference/env_pi0fast.py` for real-time control and visualization in the MuJoCo environment.
- **Performance Comparison:**  
  Use `utils/show_inference.py` to compare MAE/MSE and action sequences between pi0 and pi0-FAST.

*Leave space for additional demo GIFs or screenshots here.*

---

## 📈 Project Flow (Paper Figure Placeholder)

*Leave space for a high-level project flowchart or system diagram here. (To be added in documentations/)*

---

## 📄 Main Reference

- **FAST: Efficient Action Tokenization for Vision-Language-Action Models**  
  https://arxiv.org/abs/2501.09747

- **π₀: A Vision-Language-Action Flow Model for General Robot Control**  
  https://arxiv.org/abs/2410.24164  

---

## 🛠️ Troubleshooting

- It is recommended to use `requirements.txt` or `environment.yml` to quickly set up the project environment:
  - For pip users: `pip install -r requirements.txt`
  - For conda users: `conda env create -f environment.yml`
- All scripts should be run from the project root directory to ensure correct relative paths.
- If you encounter MuJoCo or Gymnasium related issues, please refer to their official documentation for environment setup.
- If you see ImportError or missing dependencies, install the required package as indicated by the error message.
- For GPU acceleration, make sure PyTorch is installed with the appropriate CUDA version and set `device="cuda"` in your scripts.
- For more details and common issues, please refer to the README in each submodule.

---
