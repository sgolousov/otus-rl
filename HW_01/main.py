import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random

'''
https://gymnasium.farama.org/api/env/
https://gymnasium.farama.org/environments/toy_text/taxi/
'''

def e_greedy_policy(q, epsilon):
    result = 0

    num_actions = len(q)

    if random.random() < epsilon:
        result = random.randint(0, num_actions - 1)
    else:
        # max_value = max(q)
        # indices = np.where(q == max_value)[0]
        # max_value_index = random.randint(0, len(indices) - 1)
        # result = max_value_index
        result = np.argmax(q)

    return result


def q_learning(env, policy, alpha, gamma, num_episodes):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    Q = np.zeros((num_states, num_actions))
    total_rewards = np.zeros(num_episodes)


    for i in np.arange(num_episodes):
        index = i + 1
        epsilon = 1 / index
        state, info = env.reset()
        total_episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_prev = state

            action = policy(Q[state], epsilon)

            state, reward, terminated, truncated, info = env.step(action)

            Q[state_prev][action] += \
                alpha * (reward + gamma * max(Q[state]) - Q[state_prev][action])

            total_episode_reward += reward

        total_rewards[i] = total_episode_reward
    
    return total_rewards

def calc_sliding_window_mean(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='valid')

def main():
    env = gym.make("Taxi-v4")

    num_episodes = 1000
    mean_window_width = 100

    total_rewards = q_learning(env=env, policy=e_greedy_policy, alpha=0.5, gamma=0.999, num_episodes=num_episodes)
    total_rewards_mean = calc_sliding_window_mean(total_rewards, mean_window_width)
    
    plt.plot(np.arange(num_episodes), total_rewards, label='Reward progression')
    plt.plot(np.arange(mean_window_width - 1, num_episodes), total_rewards_mean, label='Reward progression (mean)')
    plt.title('Total episode reward')
    plt.xlabel('Index of episode')
    plt.ylabel('Total episode reward')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()