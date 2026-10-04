import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import AutoresetMode

import config
import checkpoints
from models import Actor, Critic

def preprocess_frame(frame):
    frame = np.asarray(frame, dtype=np.float32) / 255.0
    return np.transpose(frame, (2, 0, 1))

def discounted_returns(rewards, gamma):
    trajectory_rewards = np.zeros(len(rewards), dtype=np.float64)
    for j in reversed(range(len(trajectory_rewards))):
        trajectory_rewards[j] = rewards[j] + (
            trajectory_rewards[j + 1] * gamma if j + 1 < len(trajectory_rewards) else 0
        )
    return trajectory_rewards

def scheduled_entropy_coef(episode, num_episodes):
    span = max(num_episodes - 1, 1)
    progress = min(max(episode / span, 0.0), 1.0)
    start = config.ENTROPY_COEF_START
    end = config.ENTROPY_COEF_END
    return start + progress * (end - start)

def scheduled_steer_delta(episode):
    span = max(config.STEER_RATE_EPISODES, 1)
    progress = min(max(episode / span, 0.0), 1.0)
    start = config.STEER_RATE_START
    end = config.STEER_RATE_END
    return start + progress * (end - start)

def update_models(optimizer_actor, optimizer_critic, actor_model: Actor, critic_model: Critic, gamma, states, raw_actions, old_log_probs, trajectory_rewards, entropy_coef):

    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=config.device)
    trajectory_rewards = torch.as_tensor(trajectory_rewards, dtype=torch.float32, device=config.device)
    raw_actions = torch.as_tensor(np.asarray(raw_actions), dtype=torch.float32, device=config.device)
    old_log_probs = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32, device=config.device)
    
    values = critic_model(states).squeeze(1)
    vf_loss = nn.MSELoss()(
        values,
        trajectory_rewards)
    optimizer_critic.zero_grad()
    # считаем градиенты
    vf_loss.backward()
    nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
    # делаем шаг оптимизатора
    optimizer_critic.step()

    # Оптимизируем policy loss (Actor). Several passes over the same
    # trajectory let the clip reject a step that moves π too far.
    with torch.no_grad():
        values = critic_model(states).squeeze(1)
        advantages = trajectory_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_size = states.shape[0]
    minibatch_size = min(config.PPO_MINIBATCH, batch_size)
    for _ in range(config.PPO_EPOCHS):
        permutation = torch.randperm(batch_size, device=config.device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start:start + minibatch_size]
            log_probs, entropy = actor_model.log_prob(states[idx], raw_actions[idx])
            ratio = torch.exp(log_probs - old_log_probs[idx])
            surrogate = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - config.PPO_CLIP, 1.0 + config.PPO_CLIP) * advantages[idx]
            pi_loss = -torch.min(surrogate, clipped).mean() - entropy_coef * entropy.mean()

            optimizer_actor.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
            optimizer_actor.step()

    return 

def build_models(alpha):
    # Two policy outputs: steering and one throttle axis.
    actor_model = Actor(2).to(config.device)
    critic_model = Critic().to(config.device)
    optimizer_actor = optim.Adam(actor_model.parameters(), lr=config.ACTOR_LR)
    optimizer_critic = optim.Adam(critic_model.parameters(), lr=alpha)
    return actor_model, critic_model, optimizer_actor, optimizer_critic

def note_episode(total_rewards, episode, episode_reward, entropy_coef, steer_delta):
    # episode is the index of this finished episode, so the history must
    # already contain exactly the previous ones.
    total_rewards.append(float(episode_reward))
    window = total_rewards[max(0, episode - 99):episode + 1]
    print(
        "Run episode {} with reward {} entropy {:.4f} steer_delta {:.3f}".format(
            episode, total_rewards[episode], entropy_coef, steer_delta
        ),
        end="\n",
    )
    if episode % 100 == 0:
        print(
            "Run episode {} with average reward {} entropy {:.4f} steer_delta {:.3f}".format(
                episode, np.mean(window), entropy_coef, steer_delta
            ),
            end="\r",
        )

    return np.average(window) > 900.0

def frames_to_tensor(frames):
    # Vector env batch is NHWC uint8. Conv layers expect NCHW in [0, 1].
    frames = np.ascontiguousarray(frames)
    tensor = torch.from_numpy(frames).to(config.device)
    return tensor.permute(0, 3, 1, 2).contiguous().float().div(255.0)

def make_vec_envs(num_envs, vectorization_mode):
    return gym.make_vec(
        "CarRacing-v3",
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        vector_kwargs={"autoreset_mode": AutoresetMode.DISABLED},
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

def a2c_vec(num_envs, vectorization_mode, alpha, gamma, num_episodes, resume_path=None, save_every=100, checkpoint_dir="checkpoints"):
    actor_model, critic_model, optimizer_actor, optimizer_critic = build_models(alpha)
    completed, total_rewards = checkpoints.resume_training(
        resume_path, actor_model, critic_model, optimizer_actor, optimizer_critic
    )
    finished = checkpoints.finish_if_complete(completed, num_episodes, total_rewards)
    if finished is not None:
        return finished

    envs = make_vec_envs(num_envs, vectorization_mode)
    try:
        try:
            obs, _info = envs.reset()
            states = [[] for _ in range(num_envs)]
            raw_actions = [[] for _ in range(num_envs)]
            old_log_probs = [[] for _ in range(num_envs)]
            rewards = [[] for _ in range(num_envs)]
            last_steering = np.zeros(num_envs, dtype=np.float32)
            steer_delta = np.full(num_envs, scheduled_steer_delta(completed), dtype=np.float32)

            stop = False

            while completed < num_episodes and not stop:
                obs_tensor = frames_to_tensor(obs)
                with torch.no_grad():
                    env_action, raw_action, log_prob = actor_model(
                        obs_tensor,
                        torch.as_tensor(last_steering, device=config.device),
                        torch.as_tensor(steer_delta, device=config.device),
                    )
                action_batch = env_action.cpu().numpy()
                raw_batch = raw_action.cpu().numpy()
                log_prob_batch = log_prob.cpu().numpy()

                next_obs, step_rewards, terminated, truncated, _info = envs.step(action_batch)
                last_steering = np.ascontiguousarray(action_batch[:, 0], dtype=np.float32)
                done = np.logical_or(terminated, truncated)

                finished_envs = []
                for i in range(num_envs):
                    states[i].append(np.ascontiguousarray(obs[i]))
                    raw_actions[i].append(np.ascontiguousarray(raw_batch[i]))
                    old_log_probs[i].append(float(log_prob_batch[i]))
                    rewards[i].append(float(step_rewards[i]))
                    if done[i]:
                        finished_envs.append(i)

                for i in finished_envs:
                    if completed >= num_episodes:
                        break

                    processed_states = [preprocess_frame(frame) for frame in states[i]]
                    entropy_coef = scheduled_entropy_coef(completed, num_episodes)
                    update_models(
                        optimizer_actor,
                        optimizer_critic,
                        actor_model,
                        critic_model,
                        gamma,
                        processed_states,
                        raw_actions[i],
                        old_log_probs[i],
                        discounted_returns(rewards[i], gamma),
                        entropy_coef,
                    )

                    stopped = note_episode(
                        total_rewards, completed, sum(rewards[i]), entropy_coef, float(steer_delta[i])
                    )
                    checkpoints.save_progress(
                        checkpoint_dir,
                        total_rewards,
                        actor_model,
                        critic_model,
                        optimizer_actor,
                        optimizer_critic,
                        save_every,
                        force=stopped,
                    )

                    completed += 1
                    states[i].clear()
                    raw_actions[i].clear()
                    old_log_probs[i].clear()
                    rewards[i].clear()
                    if stopped:
                        stop = True
                        break

                if np.any(done):
                    next_obs, _info = envs.reset(options={"reset_mask": np.asarray(done, dtype=np.bool_)})
                    last_steering[done] = 0.0
                    steer_delta[done] = scheduled_steer_delta(completed)

                obs = next_obs
        except KeyboardInterrupt:
            checkpoints.report_interrupt(
                checkpoints.save_progress(
                    checkpoint_dir,
                    total_rewards,
                    actor_model,
                    critic_model,
                    optimizer_actor,
                    optimizer_critic,
                    save_every,
                    force=True,
                ),
                total_rewards,
            )
    finally:
        envs.close()

    return np.asarray(total_rewards, dtype=np.float64)
