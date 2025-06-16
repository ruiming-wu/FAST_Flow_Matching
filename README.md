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

## 📁 Repository Structure

This repository is organized into several main modules, each responsible for a specific aspect of the project:

- **data/**  
  Collected trajectory data, parameters, and visualizations.  

- **data_collection/**  
  Scripts and utilities for generating and saving control trajectories using PID controllers.  

- **inference/**  
  Code for running inference and evaluation with trained models. 

- **model/**  
  Model definitions, including Transformer architectures for sequence modeling and flow matching.  

- **train/**  
  Training scripts, logs, and checkpoints for model optimization and evaluation.  

- **utils/**  
  Helpful scripts.

Each module contains a dedicated README with further explanations and usage instructions.
