import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config

def calc_log_probs(dist, raw_action):
    # log π(tanh(u)). The jacobian equals log(1 - tanh(u)^2), written so it
    # stays finite for large u instead of inverting tanh on a stored action.
    log_prob = dist.log_prob(raw_action)
    log_det = 2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
    return (log_prob - log_det).sum(dim=-1)

class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 96x96 -> 23x23 -> 10x10 -> 4x4
        self.out_dim = 128 * 4 * 4

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=512):
        super(Actor, self).__init__()
        self.encoder = ConvEncoder()
        self.fc = nn.Linear(self.encoder.out_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def distribution(self, x):
        hidden = torch.relu(self.fc(self.encoder(x)))
        mu = self.mu(hidden)
        log_std = self.log_std(hidden).clamp(config.LOG_STD_MIN, config.LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def to_env_action(self, squashed, last_steering=None, max_steer_delta=None):
        # squashed is (steering, throttle) in [-1, 1]. Positive throttle is gas
        # and negative is brake, so the two pedals cannot cancel each other.
        # The steering command is clamped to the previous command ± max_steer_delta
        # before the gas cut, so the pedals follow the steering that is sent.
        steering = squashed[:, 0]
        throttle = squashed[:, 1]
        if last_steering is not None and max_steer_delta is not None:
            last = torch.as_tensor(last_steering, device=steering.device, dtype=steering.dtype)
            delta = torch.as_tensor(max_steer_delta, device=steering.device, dtype=steering.dtype)
            if last.ndim == 0:
                last = last.expand_as(steering)
            if delta.ndim == 0:
                delta = delta.expand_as(steering)
            low = (last - delta).clamp(-1.0, 1.0)
            high = (last + delta).clamp(-1.0, 1.0)
            steering = torch.max(torch.min(steering, high), low)
            
        # Gas is dropped once |steering| passes STEER_GAS_THRESHOLD: a sharp
        # turn with the accelerator on slides the car off the track.
        gas = throttle.clamp(min=0.0) * config.GAS_SCALE
        gas = gas.masked_fill(steering.abs() > config.STEER_GAS_THRESHOLD, 0.0)
        brake = (-throttle).clamp(min=0.0) * config.BRAKE_SCALE
        return torch.stack([steering, gas, brake], dim=-1)

    def forward(self, x, last_steering=None, max_steer_delta=None):
        dist = self.distribution(x)
        raw_action = dist.sample()
        log_prob = calc_log_probs(dist, raw_action)
        env_action = self.to_env_action(torch.tanh(raw_action), last_steering, max_steer_delta)
        return env_action, raw_action, log_prob

    def log_prob(self, states, raw_actions):
        dist = self.distribution(states)
        log_probs = calc_log_probs(dist, raw_actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

class Critic(nn.Module):
    def __init__(self, hidden_dim=512):
        super(Critic, self).__init__()
        self.encoder = ConvEncoder()
        self.fc1 = nn.Linear(self.encoder.out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(self.encoder(x)))
        return self.fc2(x)
