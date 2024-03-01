# inspired from https://github.com/denisyarats/dmc2gym

import logging
import os
from gymnasium.spaces import Box, Dict
from gymnasium.core import Env
import numpy as np
from dm_control import suite
from dm_env import specs


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        # assert s.dtype == np.float64 or s.dtype == np.float32 or 
        assert np.issubdtype(s.dtype, np.number)
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=dtype)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=s.dtype)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)

    assert low.shape == high.shape
    return Box(low, high, dtype=low.dtype)

def _spec_to_box_v2(spec):
    boxes = {}
    for s in spec:
        if type(s) == specs.Array:
            s_min = np.array(-np.inf)
            s_max = np.array(np.inf)
        elif type(s) == specs.BoundedArray:
            s_min = s.minimum
            s_max = s.maximum
        if np.ndim(s_min) == 0:
            s_min = s_min.item()
        if np.ndim(s_max) == 0:
            s_max = s_max.item()
        box = Box(s_min, s_max, s.shape, s.dtype)
        if s.name:
            boxes[s.name] = box
    if len(boxes.keys()) > 1:
        return Dict(boxes)
    else:
        return next(iter(boxes.values()))


# def _spec_to_box(spec):
#     return Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCGym(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        domain,
        task,
        task_kwargs={},
        environment_kwargs={},
        rendering='egl',
        render_height=64,
        render_width=64,
        render_camera_id=0,
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        if rendering:
            assert rendering in ["glfw", "egl", "osmesa"]
            os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box_v2(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        # observation = timestep.observation
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed:
            logging.warn(
                "Currently DMC has no way of seeding episodes. It only allows to seed experiments on environment initialization"
            )

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        # observation = timestep.observation
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
