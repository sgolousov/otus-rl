import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

import config
import train

'''
https://gymnasium.farama.org/api/env/
https://gymnasium.farama.org/environments/box2d/car_racing/
'''

def calc_sliding_window_mean(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='valid')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Actor-Critic agent on CarRacing-v3")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file to continue training from")
    parser.add_argument("--save-every", type=int, default=100, help="Save a numbered checkpoint every N finished episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--require-cuda", action="store_true", help="Stop if CUDA is not available instead of training on CPU")
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    return args

def select_device(require_cuda):
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this Python. --require-cuda was set, so training did not start.\n"
            "Install a CUDA build of PyTorch, then run this file again:\n"
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cu130"
        )
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    select_device(args.require_cuda)
    num_episodes = 3000
    mean_window_width = 100
    alpha = 1e-3
    gamma = 0.99

    if config.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        num_envs = config.NUM_CUDA_ENVS
        vectorization_mode = "async"
        print(f"Training on {torch.cuda.get_device_name(config.device)} with {num_envs} environments")
    else:
        num_envs = 1
        vectorization_mode = "sync"
        print("Training on CPU")

    total_rewards = train.a2c_vec(
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        alpha=alpha,
        gamma=gamma,
        num_episodes=num_episodes,
        resume_path=args.resume,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )

    recorded = np.asarray(total_rewards, dtype=np.float64)
    n = len(recorded)
    plt.plot(np.arange(n), recorded, label='Reward progression')
    if n >= mean_window_width:
        total_rewards_mean = calc_sliding_window_mean(recorded, mean_window_width)
        plt.plot(np.arange(mean_window_width - 1, n), total_rewards_mean, label='Reward progression (mean)')
    plt.title('Total episode reward')
    plt.xlabel('Index of episode')
    plt.ylabel('Total episode reward')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
