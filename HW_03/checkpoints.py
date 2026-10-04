import shutil
from pathlib import Path

import numpy as np
import torch

import config

def checkpoint_state(episode, actor_model, critic_model, optimizer_actor, optimizer_critic, total_rewards):
    return {
        "episode": episode,
        "actor": actor_model.state_dict(),
        "critic": critic_model.state_dict(),
        "optimizer_actor": optimizer_actor.state_dict(),
        "optimizer_critic": optimizer_critic.state_dict(),
        "total_rewards": [float(reward) for reward in total_rewards],
    }

def write_checkpoints(checkpoint_dir, episode, actor_model, critic_model, optimizer_actor, optimizer_critic, total_rewards):
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numbered = directory / f"ep_{episode:04d}.pt"
    latest = directory / "latest.pt"
    torch.save(
        checkpoint_state(episode, actor_model, critic_model, optimizer_actor, optimizer_critic, total_rewards),
        numbered,
    )
    shutil.copy2(numbered, latest)
    noun = "episode" if episode == 1 else "episodes"
    print(f"Saved checkpoint after {episode} {noun} to {numbered} and {latest}")

def load_checkpoint(path, actor_model, critic_model, optimizer_actor, optimizer_critic):
    try:
        ckpt = torch.load(path, map_location=config.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=config.device)
    actor_model.load_state_dict(ckpt["actor"])
    critic_model.load_state_dict(ckpt["critic"])
    optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
    optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
    total_rewards = [float(reward) for reward in ckpt["total_rewards"]]
    return int(ckpt["episode"]), total_rewards

def resume_training(resume_path, actor_model, critic_model, optimizer_actor, optimizer_critic):
    if resume_path is None:
        return 0, []
    start_episode, total_rewards = load_checkpoint(
        resume_path, actor_model, critic_model, optimizer_actor, optimizer_critic
    )
    if len(total_rewards) != start_episode:
        print(
            "Checkpoint episode {} does not match {} stored rewards; continuing from {}".format(
                start_episode, len(total_rewards), len(total_rewards)
            )
        )
        start_episode = len(total_rewards)
    print(f"Resumed from {resume_path} at episode {start_episode}")
    return start_episode, total_rewards

def save_progress(checkpoint_dir, total_rewards, actor_model, critic_model, optimizer_actor, optimizer_critic, save_every, force=False):
    episode = len(total_rewards)
    if episode <= 0 or (not force and episode % save_every != 0):
        return False
    write_checkpoints(
        checkpoint_dir, episode, actor_model, critic_model, optimizer_actor, optimizer_critic, total_rewards
    )
    return True

def finish_if_complete(start_episode, num_episodes, total_rewards):
    if start_episode < num_episodes:
        return None
    print(f"Checkpoint already has {start_episode} episodes; num_episodes is {num_episodes}.")
    return np.asarray(total_rewards, dtype=np.float64)

def report_interrupt(saved, total_rewards):
    if saved:
        noun = "episode" if len(total_rewards) == 1 else "episodes"
        print(f"Training interrupted. Saved checkpoint after {len(total_rewards)} {noun}.")
    else:
        print("Training interrupted before any episode finished.")
