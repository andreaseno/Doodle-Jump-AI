from gym.envs.registration import register

register(
    id='doodlejump-v0',
    entry_point='gym_doodlejump.envs:DoodleJumpEnv',
)
