import gymnasium as gym
import dsrl

# Create the environment
env = gym.make('OfflineCarCircle1Gymnasium-v0')

# Each task is associated with a dataset
# dataset contains observations, next_observatiosn, actions, rewards, costs, terminals, timeouts
dataset = env.get_dataset()
print(dataset['observations']) # An N x obs_dim Numpy array of observations

# dsrl abides by the OpenAI gym interface
obs, info = env.reset()
obs, reward, terminal, timeout, info = env.step(env.action_space.sample())
cost = info["cost"]

# Apply dataset filters [optional]
# dataset = env.pre_process_data(dataset, filter_cfgs)
