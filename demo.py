import os
import argparse
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from inference.inference_pi0 import infer_pi0_action_sequence
from inference.inference_pi0fast import infer_pi0fast_token_sequence
from fast.decoder import decoder
from scipy.fftpack import idct
from utils.build_corpus import GAMMA

def reset_env(env):
    obs, _ = env.reset()
    if hasattr(env.unwrapped, 'state'):
        env.unwrapped.state = np.zeros_like(env.unwrapped.state)
        obs, _ = env.reset()
    return obs

def get_model_path(model_type, selection, model_dir="train/trained_models"):
    prefix = "tinypi0fast" if model_type == "pi0fast" else "tinypi0"
    files = sorted([f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pth")])
    if not files:
        raise FileNotFoundError(f"No model files found for prefix {prefix} in {model_dir}")
    if not (1 <= selection <= len(files)):
        raise ValueError(f"Selection {selection} out of range for {prefix}, found {len(files)} models.")
    return os.path.join(model_dir, files[selection - 1])

def info_print(args, msg):
    print(f"[{args.model}] {msg}")

def main():
    parser = argparse.ArgumentParser(description="Demo for pi0/pi0fast interaction with InvertedPendulum-v5")
    parser.add_argument('--model', '-m', choices=['pi0', 'pi0fast'], required=True, help="Model type: pi0 or pi0fast")
    parser.add_argument('--selection', '-s', type=int, required=True, help="Model selection index (1-10)")
    parser.add_argument('--tokenizer-path', type=str, default="fast/tokenizer/fast_tokenizer.json", help="Path to tokenizer (for pi0fast)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', type=bool, default=True, help="Render environment")
    parser.add_argument('--chunk-len', type=int, default=50)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--replan-interval', type=int, default=10)
    parser.add_argument('--time-sleep', type=float, default=0.02)
    args = parser.parse_args()

    model_path = get_model_path(args.model, args.selection)
    info_print(args, f"Selected model: {model_path}")

    env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0, render_mode="human" if args.render else None)
    obs = reset_env(env)

    step_since_perturb = 0
    try:
        while True:
            if step_since_perturb >= 100:
                sign = np.random.choice([-1, 1])
                perturb = sign * np.random.uniform(1, 2)
                obs, _, terminated, truncated, _ = env.step([perturb])
                info_print(args, f"Perturbation applied: {perturb:.3f}")
                step_since_perturb = 0
                if args.render:
                    env.render()
                if terminated:
                    info_print(args, "Terminated, resetting environment.")
                    obs = reset_env(env)
                    continue
                elif truncated:
                    info_print(args, "Truncated (out of 1000 steps), resetting environment.")
                    obs = reset_env(env)
                    continue
                elif abs(obs[0]) > 5:
                    info_print(args, "Perturbed Cart out of [-5,5], resetting environment.")
                    obs = reset_env(env)
                    continue

            state_vec = obs[:4]
            if args.model == 'pi0':
                actions = infer_pi0_action_sequence(
                    model_path, state_vec, chunk_len=args.chunk_len, device=args.device
                )
                actions = actions.squeeze(-1) if actions.ndim > 1 else actions
            else:
                tokens = infer_pi0fast_token_sequence(
                    model_path, state_vec, max_seq_len=args.max_seq_len, device=args.device
                )
                token_ints = decoder(tokens, args.tokenizer_path)
                if len(token_ints) < args.chunk_len:
                    token_ints = token_ints + [0] * (args.chunk_len - len(token_ints))
                token_ints = token_ints[:args.chunk_len]
                quantized = np.array(token_ints) / GAMMA
                actions = idct(quantized, norm='ortho')[:args.chunk_len]

            for i in range(args.replan_interval):
                action = np.array([actions[i]])
                obs, _, terminated, truncated, _ = env.step(action)
                step_since_perturb += 1
                if args.render:
                    env.render()
                if terminated:
                    info_print(args, "Terminated, resetting environment.")
                    obs = reset_env(env)
                    step_since_perturb = 0
                    continue
                elif truncated:
                    info_print(args, "Truncated (out of 1000 steps), resetting environment.")
                    obs = reset_env(env)
                    continue
                elif abs(obs[0]) > 5:
                    info_print(args, "Controlled Cart out of [-5,5], resetting environment.")
                    obs = reset_env(env)
                    step_since_perturb = 0
                    break
                time.sleep(args.time_sleep)
    except KeyboardInterrupt:
        info_print(args, "Demo interrupted by user. Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()