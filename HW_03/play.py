import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

import config
import train
from models import Actor

ACTION_NAMES = ("steering", "gas", "brake")
ACTION_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
ACTION_LIMITS = ((-1.05, 1.05), (-0.05, 1.05), (-0.05, 1.05))


def parse_args():
    parser = argparse.ArgumentParser(description="Play a CarRacing-v3 episode from an actor checkpoint, then render it to a video")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Actor checkpoint to load")
    parser.add_argument("--video-dir", type=str, default="videos", help="Folder where the episode video is written")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to record")
    parser.add_argument("--seed", type=int, default=None, help="Reset seed for the first episode; later episodes use seed+1, seed+2, ...")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions the way training does; otherwise use the mean action")
    parser.add_argument("--action-graph", action="store_true", help="Open a steering, gas, brake, and reward plot after each episode")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    return args

def load_actor(path):
    actor = Actor(2).to(config.device)
    try:
        ckpt = torch.load(path, map_location=config.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=config.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, int(ckpt["episode"])

def select_action(actor, frame, stochastic):
    state = torch.as_tensor(train.preprocess_frame(frame), device=config.device).unsqueeze(0)
    with torch.no_grad():
        if stochastic:
            env_action, _raw_action, _log_prob = actor(state)
        else:
            env_action = actor.to_env_action(torch.tanh(actor.distribution(state).mean))
    return env_action.squeeze(0).cpu().numpy()

def make_env(render_mode):
    return gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

def show_action_graph(actions, rewards, episode):
    actions = np.asarray(actions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"expected actions with shape (steps, 3), got {actions.shape}")
    if rewards.shape != (actions.shape[0],):
        raise ValueError(f"expected one reward per step, got {rewards.shape} for {actions.shape[0]} steps")
    steps = np.arange(actions.shape[0])
    _fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    for index, ax in enumerate(axes[:3]):
        if index == 0:
            ax.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.plot(steps, actions[:, index], color=ACTION_COLORS[index])
        ax.set_ylim(*ACTION_LIMITS[index])
        ax.set_ylabel(ACTION_NAMES[index])
        ax.grid(True)
    reward_ax = axes[3]
    reward_ax.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
    reward_ax.plot(steps, rewards, color="#ff7f0e")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(True)
    axes[0].set_title(f"Episode {episode} actions (reward {rewards.sum():.1f})")
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.show()

def make_video_env(video_dir):
    return RecordVideo(
        make_env("rgb_array"),
        video_folder=video_dir,
        episode_trigger=lambda episode: True,
        name_prefix="carracing",
    )

def rollout_episode(actor, env, stochastic, seed):
    state, _info = env.reset(seed=seed)
    actions = []
    rewards = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = select_action(actor, state, stochastic)
        actions.append(action.copy())
        state, reward, terminated, truncated, _info = env.step(action)
        rewards.append(float(reward))
    return actions, rewards

def render_episode(env, actions, seed):
    env.reset(seed=seed)
    for step, action in enumerate(actions, start=1):
        _state, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            if step != len(actions):
                print(f"Replay stopped at step {step} of {len(actions)}")
            break
    else:
        print(f"Replay used all {len(actions)} actions without the episode ending")
    if env.recording:
        env.stop_recording()

def main():
    args = parse_args()
    actor, trained_episodes = load_actor(args.checkpoint)
    print(f"Loaded actor from {args.checkpoint} after {trained_episodes} episodes")
    print(f"Playing on {config.device}")

    play_env = make_env(None)
    video_env = make_video_env(args.video_dir)
    video_dir = Path(args.video_dir).resolve()
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode if args.seed is not None else random.randrange(2**31)
            print(f"Episode {episode}: playing")
            actions, rewards = rollout_episode(actor, play_env, args.stochastic, seed)
            print(f"Episode {episode} reward {sum(rewards):.1f} over {len(actions)} steps (seed {seed})")
            print(f"Episode {episode}: rendering")
            render_episode(video_env, actions, seed)
            print(f"Saved {video_dir / f'carracing-episode-{episode}.mp4'}")
            if args.action_graph:
                show_action_graph(actions, rewards, episode)
    finally:
        play_env.close()
        video_env.close()


if __name__ == "__main__":
    main()
