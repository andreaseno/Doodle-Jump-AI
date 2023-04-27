#ex.py

import gym
import gym_doodlejump

env = gym.make('doodlejump-v0', render_mode='human')

observation, info = env.reset()
action = env.action_space.sample()
for i in range(10000):
    if i % 10 == 0:
        action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
