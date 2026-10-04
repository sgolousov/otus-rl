import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State-dependent noise stays inside this range. Below -5 the squashed
# policy is effectively deterministic and the run cannot leave a bad action.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTOR_LR = 3e-4
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_MINIBATCH = 256
# High early entropy keeps throttle noise large enough to sample braking.
# It decays linearly to the floor across the planned number of episodes.
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END = 0.01
NUM_CUDA_ENVS = 16
# Gas is zero while |steering| is above this. Steering lives in [-1, 1],
# so a value of 1.0 or more leaves gas unchanged.
STEER_GAS_THRESHOLD = 1.0
GAS_SCALE = 0.2
BRAKE_SCALE = 0.8
# Max |change| in the steering command per step, relative to the previous
# command. Steering lives in [-1, 1], so a delta of 2 can reach any command
# from any previous one and the limit no longer binds. The allowed delta
# grows linearly from START at episode 0 to END at STEER_RATE_EPISODES.
STEER_RATE_START = 0.01
STEER_RATE_END = 2.0
STEER_RATE_EPISODES = 500
