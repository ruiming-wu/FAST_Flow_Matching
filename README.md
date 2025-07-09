# FAST-Flow-Matching

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

**Examples:**  
Below are example visualizations generated by `utils/show_inference.py` for different data chunks. These plots show the original action sequence and the predictions from both pi0 and pi0-FAST models, along with their MAE/MSE metrics.

<div align="center">
  <img src="documentations/chunk 327.png" alt="Chunk 327 Example" style="width: 100%; max-width: 900px;">
  <img src="documentations/chunk 802.png" alt="Chunk 802 Example" style="width: 100%; max-width: 900px;">
  <img src="documentations/chunk 10000.png" alt="Chunk 10000 Example" style="width: 100%; max-width: 900px;">
  <img src="documentations/chunk 99999.png" alt="Chunk 99999 Example" style="width: 100%; max-width: 900px;">
</div>

*Each figure above is produced by running `python -m utils.show_inference` with a different chunk index and demonstrates the model's prediction capability and error metrics.*

---

## 📈 Project Flow

<div align="center">
  <img src="documentations/flow chart.png" alt="Project Flowchart" style="width: 100%; max-width: 900px;">
</div>

*The above flowchart illustrates the overall data processing, model training, and evaluation pipeline for both pi0 and pi0-FAST methods in this project.*

---

## 📄 Main Reference

- **FAST: Efficient Action Tokenization for Vision-Language-Action Models**  
  https://arxiv.org/abs/2501.09747

- **π₀: A Vision-Language-Action Flow Model for General Robot Control**  
  https://arxiv.org/abs/2410.24164  

---

## 📊 More Results

### Hyperparameter Selection in FAST: The Role of Gamma

In the FAST method, the hyperparameter **gamma** controls the scaling factor during the quantization step before BPE tokenization. The choice of gamma presents a trade-off:

- **Larger gamma**:  
  - Makes it less likely for information to be lost during the round step, preserving more detail in the quantized action representation.
  - However, this also increases the size of the base vocabulary for BPE, which can lead to higher computational costs and memory usage during both tokenizer training and model inference.

- **Smaller gamma**:  
  - Reduces the vocabulary size and computational burden.
  - But may result in more information loss during quantization, potentially degrading model performance.

Below is an example trade-off curve illustrating this effect:

<div align="center">
  <img src="documentations/gamma_tradeoff.png" alt="Gamma Tradeoff Example" style="width: 100%; max-width: 700px;">
</div>

*As gamma increases, information loss decreases but the BPE base vocabulary size grows rapidly, increasing computational cost.*

---

### Tokenization Statistics with gamma=10, vocab_size=256

When using **gamma=10** and setting the BPE vocabulary size to **256**, we can analyze the compression and representation efficiency of the FAST method on the entire dataset:

- **Token Sequence Length Distribution:**  
  The histogram below shows the distribution of tokenized sequence lengths (excluding special tokens) for all samples in the training set.  
  - **Average token sequence length:** 7.66

<div align="center">
  <img src="documentations/token seq length.png" alt="Token Sequence Length Distribution" style="width: 100%; max-width: 700px;">
</div>

- **Vocabulary Token Length Distribution:**  
  The histogram below shows the length (number of elements) of each token in the vocabulary.  
  - **Average token length:** 5.43

<div align="center">
  <img src="documentations/vocab token length.png" alt="Vocab Token Length Distribution" style="width: 100%; max-width: 700px;">
</div>

*These results demonstrate that with gamma=10 and vocab_size=256, the FAST method achieves significant sequence compression while maintaining a manageable average token length in the vocabulary.*

---

### Control Performance Comparison: pi0 vs pi0-FAST

To comprehensively evaluate the control performance of both models, we consider four key metrics under various noise and replan interval settings:

1. **Success Rate:**  
   The percentage of episodes where the agent is able to keep the CartPole upright for the full 100 steps (i.e., no termination or truncation occurs before 100 steps in the simulation environment).

2. **Converge Rate:**  
   The percentage of episodes where, within the 100 steps, there exists a sequence of 10 consecutive steps such that the absolute value of the pole angle at each of these steps is less than 0.02 radians (indicating convergence).

3. **Average Converge Steps:**  
   For episodes that are both successful and convergent, this metric records the average step at which convergence is first achieved.

4. **Average Velocity Standard Deviation:**  
   Measures the standard deviation of the cart's velocity over the episode, reflecting how well the controller stabilizes the velocity.

#### Experimental Settings

- **Noise Level:**  
  Three levels are tested: 0.01, 0.03, 0.05 (the dataset was generated under 0.05 noise using PID).
- **Replan Interval:**  
  The controller replans every 10, 13, or 15 steps (although each model predicts 50 steps per rollout).
- For each combination of noise level and replan interval (9 settings in total), 1000 episodes are run for each model (pi0 and pi0-FAST), using 10 different checkpoints per model (100 episodes per checkpoint), totaling **18,000 simulations**.

#### Results

<div align="center">
  <img src="documentations/Success Rate.png" alt="Success Rate" style="width: 100%; max-width: 700px;">
  <img src="documentations/Converge Rate.png" alt="Converge Rate" style="width: 100%; max-width: 700px;">
  <img src="documentations/Average Converge Steps.png" alt="Average Converge Steps" style="width: 100%; max-width: 700px;">
  <img src="documentations/Cart Velocity Standard Deviation.png" alt="Average Velocity Standard Deviation" style="width: 100%; max-width: 700px;">
</div>

**Observations:**  
- **pi0** generally outperforms **pi0-FAST** in terms of convergence rate and converging quickly, indicating better ability to stabilize the pole angle.
- However, **pi0-FAST** achieves a lower average velocity standard deviation, meaning it controls the cart's velocity more smoothly and stably.
- This may be due to the frequency-domain compression in FAST: high-frequency (rapid) changes in the action sequence, which are important for fine angle control, may be partially lost during quantization and tokenization. In contrast, the flow matching approach in pi0 preserves these high-frequency details, leading to better convergence but potentially rougher velocity control.
- The trade-off between angle convergence and velocity smoothness is evident, and may be influenced by the underlying PID control structure and the information bottleneck introduced by tokenization.

---

### Training & Inference Efficiency Comparison: pi0 vs pi0-FAST

Below are the metric comparison plots for training and inference efficiency:

<div align="center">
  <img src="documentations/training_metrics.png" alt="Training-related Metrics Comparison" style="width: 100%; max-width: 700px;">
  <img src="documentations/inference_metrics.png" alt="Inference-related Metrics Comparison" style="width: 100%; max-width: 700px;">
</div>

In addition to control performance, we compare **training-related** and **inference-related** metrics between pi0 and pi0-FAST, as shown in the figures above.

#### Training-related Metrics

- **Dataset Size (MB):**  
  pi0-FAST achieves a much smaller dataset size (12.50 MB) compared to pi0 (41.10 MB), demonstrating the strong compression capability of the FAST tokenization method.
- **Average Sequence Length:**  
  The average tokenized sequence length for pi0-FAST is only 7.66, while pi0 uses the full action sequence length of 50. This highlights the efficiency of the quantization and BPE-based compression in FAST.
- **Training Time (min):**  
  Despite the smaller dataset and shorter sequences, pi0-FAST requires more training time (53.48 min) than pi0 (42.34 min). This may be due to the increased complexity of the token prediction task and the overhead of handling discrete token sequences.

#### Inference-related Metrics

- **Model Preparation Time (ms):**  
  pi0-FAST takes longer to load and prepare the model (32.84±4.22 ms) compared to pi0 (24.55±3.58 ms).
- **Inference Time (ms):**  
  pi0-FAST has a higher average inference time per step (3.80±0.59 ms for one token generation) than pi0 (2.36±0.47 ms for one step vector field generation).
- **Action Reconstruction Time (ms):**  
  The time to reconstruct the full action sequence from tokens is significantly higher for pi0-FAST (56.26±20.01 ms) than for pi0 (24.67±4.83 ms), indicating that the decoding and dequantization process introduces additional latency.

#### Summary

- **Compression:**  
  FAST provides substantial compression in both dataset size and sequence length, which is beneficial for storage and potentially for data transfer.
- **Efficiency Trade-off:**  
  The compression comes at the cost of increased training and inference time, especially in the action reconstruction phase. This suggests a trade-off between storage efficiency and computational efficiency.

These results complement the control performance analysis and provide a more comprehensive view of the strengths and limitations of the FAST approach compared to direct flow matching.

---

## 🛠️ Troubleshooting

- It is recommended to use `requirements.txt` or `environment.yml` to quickly set up the project environment:
  - For pip users: `pip install -r requirements.txt`
  - For conda users: `conda env create -f environment.yml`
- All scripts should be run from the project root directory to ensure correct relative paths.
- If you encounter MuJoCo or Gymnasium related issues, please refer to their official documentation for environment setup (some of the environment settings should be modified for larger range of control, [data_collection/README.md](./data_collection/README.md) for details).
- If you see ImportError or missing dependencies, install the required package as indicated by the error message.
- For GPU acceleration, make sure PyTorch is installed with the appropriate CUDA version and set `device="cuda"` in your scripts.
- For more details and common issues, please refer to the README in each submodule.

---
