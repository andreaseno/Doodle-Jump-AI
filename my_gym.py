#ex.py

import gym
import gym_doodlejump
import time

env = gym.make('doodlejump-v0', render_mode='human')

observation, info = env.reset()
action = env.action_space.sample()

i = 0
env.start_time = time.time()
while True:
    if i % 20 == 0:
        action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    i+=1