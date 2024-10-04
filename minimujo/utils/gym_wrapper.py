from typing import Type
import gym
import gym.spaces
import gymnasium
import numpy as np

def gymnasium_space_to_gym(space: gymnasium.spaces.Space, dtype: Type = None):
    if isinstance(space, gymnasium.spaces.Box):
        # gym Box doesn't like shape and low/high being vector
        shape = space.shape if np.isscalar(space.low) else None
        return gym.spaces.Box(space.low, space.high, shape=shape, dtype=dtype or space.dtype)
    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(space.n, start=space.start)
    if isinstance(space, gymnasium.spaces.Dict):
        spaces = {key:gymnasium_space_to_gym(val) for key, val in space.spaces.items()}
        return gym.spaces.Dict(spaces)
    if isinstance(space, gymnasium.spaces.Tuple):
        spaces = tuple([gymnasium_space_to_gym(val, dtype) for val in space.spaces])
        return gym.spaces.Tuple(spaces)
    if isinstance(space, gymnasium.spaces.MultiBinary):
        return gym.spaces.MultiBinary(space.n)
    if isinstance(space, gymnasium.spaces.Text):
        return gym.spaces.Text(max_length=space.max_length, min_length=space.min_length, charset=space.character_set)
    # As a fallback, just return the unaltered space.
    return space

def unwrap_env(env: gym.Env, cls: Type):
    if isinstance(env, GymnasiumToGymWrapper):
            env = env._env
    while env is not env.unwrapped:
        if isinstance(env, cls):
            return env
        if isinstance(env, GymnasiumToGymWrapper):
            env = env._env
        else:
            env = env.env
    if isinstance(env, cls):
            return env
    return None

class GymnasiumToGymWrapper(gym.Env):

    def __init__(self, env: gymnasium.Env, obs_dtype: Type = None):
        self._env = env

        self.metadata = env.metadata
        self.observation_space = gymnasium_space_to_gym(env.observation_space, obs_dtype)
        self.action_space = gymnasium_space_to_gym(env.action_space)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return obs
    
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
    
    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action)
        return obs, rew, term or trunc, info
    
    def __getattr__(self, name):
        return self._env.__getattr__(name)
    
    @property
    def full_unwrapped(self):
        return self._env.unwrapped