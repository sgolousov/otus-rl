import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

'''
https://gymnasium.farama.org/api/env/
https://gymnasium.farama.org/environments/box2d/lunar_lander/
'''

def e_greedy_policy(q, epsilon):
    result = 0

    num_actions = len(q)

    if random.random() < epsilon:
        result = random.randint(0, num_actions - 1)
    else:
        result = np.argmax(q)

    return result

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def is_ready(self):
        return len(self.buffer) >= self.batch_size

    def push(self, state_prev, action, reward, state, is_final):
        self.buffer.append((state_prev, action, reward, state, is_final))

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        return zip(*batch)

def update_models(optimizer, replay_buffer: ReplayBuffer, prediction_model: DQN, target_model: DQN, gamma):
    if replay_buffer.is_ready():
        states_prev, actions, rewards, states, is_finals = replay_buffer.get_batch()
        states_prev_tensor = torch.FloatTensor(states_prev)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        states_tensor = torch.FloatTensor(states)
        is_finals_tensor = torch.FloatTensor(is_finals).unsqueeze(1)

        # Update targets
        q_prev = prediction_model(states_prev_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            q = target_model(states_tensor)
            q_max = q.max(1, keepdim=True)[0]
            y = rewards_tensor + gamma * q_max * (1 - is_finals_tensor)

        # Get loss
        loss = nn.MSELoss()(q_prev, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

def deep_q_learning(env, policy, alpha, gamma, num_episodes, use_soft_update=False, tau=0.99):
    state_dim = env.observation_space.high.size
    actions_dim = env.action_space.n

    buffer_size = 50000
    batch_size = 64

    input_dim = state_dim
    hidden_dim = 128
    output_dim = actions_dim
    
    # DNN for prediction, updated every batch
    prediction_model = DQN(input_dim, hidden_dim, output_dim)
    # DNN for target, updated rarely
    target_model = DQN(input_dim, hidden_dim, output_dim)

    steps_to_update_target_model = 100
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    optimizer = optim.Adam(prediction_model.parameters(), lr=alpha)
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    total_rewards = np.zeros(num_episodes)
    for i in np.arange(num_episodes):
        index = i + 1
        state, info = env.reset()
        total_episode_reward = 0
        terminated = False
        truncated = False

        j = 0
        while not (terminated or truncated):
            state_prev = state

            # Select action from state using DQN
            tensor = torch.from_numpy(state).float()
            input_tensor = tensor.unsqueeze(0)
            q_tensor = prediction_model(input_tensor)
            q_np = q_tensor.detach().cpu().numpy()
            action = policy(q_np, epsilon)

            state, reward, terminated, truncated, info = env.step(action)

            # Save (s, a, r, s', is_done) to replay buffer
            is_done = terminated or truncated
            replay_buffer.push(state_prev, action, reward, state, is_done)

            total_episode_reward += reward
            j += 1

            update_models(optimizer, replay_buffer, prediction_model, target_model, gamma)

            if use_soft_update:
                theta_prediction = prediction_model.state_dict()
                theta_target = target_model.state_dict()

                with torch.no_grad():
                    for key in theta_target:
                        theta_target[key].copy_(tau * theta_target[key] + (1 - tau) * theta_prediction[key])
            else:
                if j % steps_to_update_target_model == 0:
                    target_model.load_state_dict(prediction_model.state_dict())

        total_rewards[i] = total_episode_reward
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if index % 10 == 1:
            print("Step ", index, " of ", num_episodes, ".")
    
    return total_rewards

def calc_sliding_window_mean(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='valid')

def main():
    env = gym.make("LunarLander-v3")

    num_episodes = 1000
    mean_window_width = 100

    total_rewards = deep_q_learning(env=env, policy=e_greedy_policy, alpha=1e-3, gamma=0.99, num_episodes=num_episodes, use_soft_update=True, tau=0.99)
    total_rewards_mean = calc_sliding_window_mean(total_rewards, mean_window_width)
    env.close()
    
    plt.plot(np.arange(num_episodes), total_rewards, label='Reward progression')
    plt.plot(np.arange(mean_window_width - 1, num_episodes), total_rewards_mean, label='Reward progression (mean)')
    plt.title('Total episode reward')
    plt.xlabel('Index of episode')
    plt.ylabel('Total episode reward')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
