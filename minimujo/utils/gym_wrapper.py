import gym
import gym.spaces
import gymnasium
import numpy as np

def gymnasium_box_to_gym(box: gymnasium.spaces.Box):
    # gym Box doesn't like shape and low/high being vector
    shape = box.shape if np.isscalar(box.low) else None
    return gym.spaces.Box(box.low, box.high, shape, box.dtype)

class GymnasiumToGymWrapper(gym.Env):

    def __init__(self, env: gymnasium.Env):
        self._env = env

        self.metadata = env.metadata
        self.observation_space = gymnasium_box_to_gym(env.observation_space)
        self.action_space = gymnasium_box_to_gym(env.action_space)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return obs
    
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
    
    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action)
        return obs, rew, term or trunc, info