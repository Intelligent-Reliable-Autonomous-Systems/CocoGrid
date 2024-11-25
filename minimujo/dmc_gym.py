# inspired from https://github.com/denisyarats/dmc2gym

import logging
import os
from gymnasium.spaces import Box, Dict
from gymnasium.core import Env
import numpy as np
from dm_control import suite
from dm_env import specs

from minimujo.minimujo_arena import MinimujoArena
from minimujo.state.minimujo_state import MinimujoState
from minimujo.state.observation import get_observation_spec

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
    
def _spec_to_box_v3(spec, image_format=None, dtype=np.float32, flatten_threshold=100):
    box_params = {}
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
        box = (s_min, s_max, s.shape, s.dtype)
        if len(s.shape) == 3 and image_format == '0-1':
            box = (0, 1, s.shape, np.float32)
        if s.name:
            box_params[s.name] = box
    
    def ensure_shape_n(val, n):
        val = np.atleast_1d(val)
        if val.shape[0] == 1:
            return np.tile(val, n)
        return val
    
    boxes = {}
    single_dim_min = np.array([])
    single_dim_max = np.array([])
    single_dim_dtype = np.int8
    range_mapping = {}
    vector_dim = 0

    for name, box in box_params.items():
        s_min, s_max, shape, dtype = box
        if len(shape) > 1:
            if np.prod(shape) < flatten_threshold:
                shape = (np.prod(shape),)
            else:
                boxes[name] = Box(*box)
                continue
        elif len(shape) == 0:
            shape = (1,)
        n = shape[0]
        if n == 0:
            continue
        s_min = ensure_shape_n(s_min, n)
        single_dim_min = np.concatenate((single_dim_min, s_min))
        s_max = ensure_shape_n(s_max, n)
        single_dim_max = np.concatenate((single_dim_max, s_max))
        range_mapping[name] = (vector_dim, vector_dim+n)
        vector_dim += n
        single_dim_dtype = np.promote_types(single_dim_dtype, dtype)
         
    if vector_dim > 0:
        boxes['vector'] = Box(single_dim_min, single_dim_max, (vector_dim,), single_dim_dtype)

    if len(boxes.keys()) > 1:
        return Dict(boxes), range_mapping, vector_dim
    else:
        return next(iter(boxes.values())), range_mapping, vector_dim


# def _spec_to_box(spec):
#     return Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.ndim(v) == 0 else v
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

def _flatten_obs_v2(obs, range_mapping, n):
    obs_copy = { key:val for key, val in obs.items() if np.atleast_1d(val).shape[0] > 0 }
    if n > 0:
        vector = np.zeros(n)
        for key, slice in range_mapping.items():
            vector[slice[0]:slice[1]] = np.atleast_1d(obs_copy.pop(key)).flatten()
        obs_copy['vector'] = vector

    if len(obs_copy.keys()) > 1:
        return obs_copy
    else:
        return next(iter(obs_copy.values()))
        

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
        image_observation_format='0-255',
        render_mode='rgb_array',
        rendering='egl',
        render_width=64,
        render_camera_id=0,
        dmc_env=None,
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        if rendering:
            assert rendering in ["glfw", "egl", "osmesa"]
            os.environ["MUJOCO_GL"] = rendering

        if dmc_env is None:
            self._env = suite.load(
                domain,
                task,
                task_kwargs,
                environment_kwargs,
            )
        else:
            self._env = dmc_env

        # placeholder to allow built in gymnasium rendering
        self.render_mode = render_mode
        self.render_height = render_width
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._action_space = _spec_to_box([self._env.action_spec()])

        self.is_image_obs = False
        self.image_observation_format = image_observation_format
        self._observation_space, self._observation_function = self._get_observation_space()

        # set seed if provided with task_kwargs
        if "random" in (task_kwargs or {}):
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)
    
    def _get_observation_space(self):
        observation_space, self.range_mapping, self.vector_dim = _spec_to_box_v3(self._env.observation_spec().values(), image_format=self.image_observation_format)
        norm_image = self.is_image_obs and self.image_observation_format == '0-1'
        def obs_func(timestep):
            observation = _flatten_obs_v2(timestep.observation, self.range_mapping, self.vector_dim)
            if norm_image:
                observation = (observation / 255.).astype(np.float32)
            return observation
        return observation_space, obs_func
        
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
        timestep = self._env.step(action)
        self.last_observation = timestep.observation
        observation = self._observation_function(timestep)

        reward = timestep.reward
        termination = hasattr(self._env._task, "should_terminate_episode") and self._env._task.should_terminate_episode(self._env._physics_proxy)
        truncation = timestep.last() and not termination
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
        self.last_observation = timestep.observation
        observation = self._observation_function(timestep)

        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        try:
            camera_id = camera_id or self.render_camera_id
            image = self._env.physics.render(height=height, width=width, camera_id=camera_id)
        except:
            if 'top_camera' in self.last_observation:
                image = self.last_observation['top_camera']
            elif 'egocentric_camera' in self.last_observation:
                image = self.last_observation['egocentric_camera']
            else:
                image = np.zeros((height, width, 3))
        return image
    
    def index_obs_as_dict(self, observation: np.ndarray, key: str):
        """Since dict observations are flattened, this gets the original values by key"""
        idx_min, idx_max = self.range_mapping.get(key, (-1, -1))
        if idx_min < 0:
            return None
        return observation[idx_min:idx_max]

class MinimujoGym(DMCGym):

    def __init__(self, observation_type=None, *args, **kwargs):
        self._observation_type = observation_type
        super().__init__(*args, **kwargs)
        self._task = self._env._task
        
        self.track_position = False
        self.trajectory = []
        self._force_backup_render = False

    def _get_observation_space(self):
        self._observation_spec = get_observation_spec(self._observation_type)
        state = self.arena.get_state_observer().get_state(self._env._physics_proxy)
        obs_space, obs_func = self._observation_spec.build_observation_space(state)
        return obs_space, lambda timestep: obs_func(self.state)
    
    @property
    def state(self) -> MinimujoState:
        """Get the raw Minimujo state representation"""
        return self.arena.current_state
    
    @property
    def task(self) -> str:
        """Get a textual description of the current episode's task"""
        return self._task.description
    
    @property
    def arena(self) -> MinimujoArena:
        """Get the internal arena object"""
        return self._task._minimujo_arena
    
    @property
    def minigrid(self) -> Env:
        """Get the minigrid environment"""
        return self._task._minimujo_arena._minigrid
    
    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width

        grid_width = self.arena._minigrid.grid.width
        grid_height = self.arena._minigrid.grid.height
        
        if self._force_backup_render:
            image = self._get_backup_image(tile_size=width//grid_width)
        else:
            try:
                camera_id = camera_id or self.render_camera_id
                image = self._env.physics.render(height=height, width=width, camera_id=camera_id)
            except:
                if 'top_camera' in self.last_observation:
                    image = self.last_observation['top_camera']
                    width, height = image.shape[:2]
                else:
                    # if OpenGL is not defined, use a backup rendering
                    self._force_backup_render = True
                    image = self._get_backup_image(tile_size=width//grid_width)

        if self.track_position:
            self._render_trajectory(image, tile_size=width//grid_width)
        return image

    def _get_backup_image(self, tile_size):
        # use a modified Minigrid rendering as a backup

        minigrid_env = self.arena._minigrid
        # render the grid, but without the agent
        image = minigrid_env.grid.render(tile_size=tile_size, agent_pos=(-100,-100), highlight_mask=None)
        # get the continuous walker position mapped to continuous grid position
        pos = self.arena.world_to_grid_positions([self.arena.walker_position])[0]
        # scale walker position and create a white box for the agent
        x, y = tuple((pos * tile_size + tile_size/2).astype(int))
        width = int(tile_size * 0.5 * 0.4)
        image[x-width:x+width,y-width:y+width] = 255
        return image
    
    def _render_trajectory(self, image, tile_size):
        # Add some points to the render

        grid_positions = self.arena.world_to_grid_positions(self.trajectory)
        color1 = np.array([28, 255, 255])
        color2 = np.array([47, 49, 146])
        max_points = 40
        skip = int(max(1, len(grid_positions) / max_points))
        for idx, pos in enumerate(grid_positions[::-skip]):
            x, y = tuple((pos * tile_size + tile_size/2).astype(int))
            width = int(tile_size * 0.5 * 0.1)
            frac = min(1, idx / max_points)
            color = (frac * color2 + (1 - frac) * color1).astype(int)
            image[x-width:x+width,y-width:y+width, :] = color