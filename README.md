# tum-adlr-ss25-12-FAST
A small-scale control modeling project comparing FAST with original action chunk / autoregressive generation with flow matching using Transformers in CartPole simulation.

## 🧭 Project Overview 
<div align="center">
  <img src="./documentations/flow chart.png" alt="Project Flowchart" style="width: 100%; max-width: 960px;"/>
</div>

## 📄 Main Reference  
- **FAST: Efficient Action Tokenization for Vision-Language-Action Models**  
  https://arxiv.org/abs/2501.09747

## 📚 Other Related Work  
- **OpenVLA: An Open-Source Vision-Language-Action Model**  
  https://arxiv.org/abs/2406.09246  
- **π₀: A Vision-Language-Action Flow Model for General Robot Control**  
  https://arxiv.org/abs/2410.24164  
- **π₀.₅: A Vision-Language-Action Model with Open-World Generalization**  
  https://arxiv.org/abs/2504.16054

## 🧪 Simulation Setup

We use the **Gymnasium MuJoCo `InvertedPendulum-v5`** environment, with custom modifications for better control and data diversity:

### Environment Modifications
- **Angle threshold**: increased from ±0.2 rad → **±0.5 rad**
- **Cart position range**: from [-1.0, 1.0] → **[-10.0, 10.0]**  

### Controller Design
- **Nested PID structure**:
  - **Outer loop**: keeps cart velocity at 0 → outputs desired pole angle ∈ [-0.3 rad, 0.3 rad]
  - **Inner loop**: minimizes angle error → outputs final force ∈ [-3 N, 3 N]
- **Base parameters** (subject to random perturbation ±10% during data collection):
  - Kp1 = -3.0, Ki1 = -0.01, Kd1 = -0.15, Kp2 = 0.08, Ki2 = 0.05, Kd2 = 0.0

### Trajectory Generation
- **Length**: 100 time steps (~2 seconds)
- **Stabilization**: pole angle `theta` typically converges to 0 within ~50 steps
- **Action noise**: final PID output perturbed by **±0.02 N** to simulate actuation uncertainty and environment noise
- **Randomized initial states**:
  - Cart position ∈ [-0.5 m, 0.5 m]
  - Cart velocity ∈ [-0.5 m/s, 0.5 m/s]
  - Pole angle ∈ [-0.5 rad, 0.5 rad]
  - Pole angular velocity ∈ [-0.5 rad/s, 0.5 rad/s]

### Storage
- Each trajectory is stored as a `.npy` file, named using a 4-digit index (e.g., `0001.npy`)
- Files are saved under the `./data/` directory

