from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from minimujo.state.minimujo_state import MinimujoState

# Indicates that a value should be inferred during the first reset()
# Can cause problems if the the number of objects, etc changes
INFER = None

class ObservationSpecification:

    AGENT_POSE_DIM = 13
    OBJECT_POSE_DIM = 13
    FEATURES_KEY = 'features'
    ARENA_KEY = 'arena'
    SCALE_KEY = 'xy_scale'
    POSE_KEY = 'pose'

    def __init__(self,
        include_walker: bool = True,
        flatten_walker: bool = True,

        include_objects: bool = True,
        flatten_objects: bool = True,
        num_objects: Optional[int] = INFER,
        shuffle_objects: bool = True,
        object_type_one_hot: Union[bool, Sequence[int]] = False,
        object_color_one_hot: Union[bool, Sequence[int]] = False,
        object_state_one_hot: Optional[int] = None,

        include_arena: bool = True,
        flatten_arena: bool = True,
        arena_size: Optional[Tuple[int, int]] = INFER,
        arena_one_hot: Union[bool, Sequence[int]] = False,
        arena_remove_border: Optional[bool] = None,
        
        include_top_camera: bool = False,
        include_xy_scale: bool = False
    ):
        self._include_walker: bool = include_walker
        self._flatten_walker: bool = flatten_walker

        self._include_objects: bool = include_objects
        self._flatten_objects: bool = flatten_objects
        self._num_objects: Optional[int] = num_objects
        # Shuffle each episode
        self._shuffle_objects: bool = shuffle_objects

        self._object_type_one_hot: Optional[np.ndarray] = None
        if object_type_one_hot is True:
            # there are 4 object types: ball, box, door, key
            self._object_type_one_hot = np.arange(4)
        elif isinstance(object_type_one_hot, Sequence):
            self._object_type_one_hot = np.asarray(object_type_one_hot)

        self._object_color_one_hot: Optional[np.ndarray] = None
        if object_color_one_hot is True:
            # there are 4 object types: ball, box, door, key
            self._object_color_one_hot = np.arange(4)
        elif isinstance(object_color_one_hot, Sequence):
            self._object_color_one_hot = np.asarray(object_color_one_hot)

        self._object_state_one_hot: Optional[np.ndarray] = None
        if object_state_one_hot is not None:
            self._object_state_one_hot = np.arange(object_state_one_hot)

        self._include_arena: bool = include_arena
        self._flatten_arena: bool = flatten_arena
        self._arena_size: Optional[Tuple[int, int]] = arena_size
        self._arena_one_hot: Optional[np.ndarray] = None
        if arena_one_hot is True:
            # there are 4 cell types: air, wall, goal, lava. But let air be empty
            self._arena_one_hot = np.arange(1, 4)
        elif isinstance(arena_one_hot, Sequence):
            self._arena_one_hot = np.asarray(arena_one_hot)
        self._arena_remove_border: bool = arena_remove_border if arena_remove_border is not None else flatten_arena
        if arena_remove_border and arena_size is not INFER:
            assert self._arena_size[0] > 2 and self._arena_size[1] > 2, f"Cannot remove border from arena of size {self._arena_size}"
            self._arena_size = self._arena_size[0]-2, self._arena_size[0]-2

        self._include_top_camera: bool = include_top_camera

        self._include_xy_scale: bool = include_xy_scale

    def build_observation_space(self, state: MinimujoState):

        if self._num_objects is INFER:
            self._num_objects = len(state.objects)

        if self._arena_size is INFER:
            self._arena_size = state.grid.shape
            if self._arena_remove_border:
                assert self._arena_size[0] > 2 and self._arena_size[1] > 2, f"Cannot remove border from arena of size {self._arena_size}"
                self._arena_size = self._arena_size[0]-2, self._arena_size[0]-2

        self._object_dim: int = ObservationSpecification.OBJECT_POSE_DIM \
            + (len(self._object_type_one_hot) if self._object_type_one_hot is not None else 1) \
            + (len(self._object_color_one_hot) if self._object_color_one_hot is not None else 1) \
            + (len(self._object_state_one_hot) if self._object_state_one_hot is not None else 1)
        
        flat_idx = 0
        flat_map = {}
        dict_spaces = {}
        dict_obs_funcs = {}

        if self._include_walker:
            pose_key = ObservationSpecification.POSE_KEY
            pose_dim = ObservationSpecification.AGENT_POSE_DIM
            if self._flatten_walker:
                obs_func = lambda state: state.pose
                flat_map[pose_key] = (flat_idx, flat_idx + pose_dim, obs_func)
                flat_idx += pose_dim
            else:
                dict_spaces[pose_key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(pose_dim,), dtype=float)
                dict_obs_funcs[pose_key] = lambda state: state.pose
            
            for key, obs in state.walker.items():
                dim = len(np.atleast_1d(obs).flatten())
                if dim == 0:
                    continue
                obs_func = lambda state, this_key = key: state.walker[this_key]
                if self._flatten_walker:
                    flat_map[key] = (flat_idx, flat_idx+dim, obs_func)
                    flat_idx += dim
                else:
                    dict_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)
                    dict_obs_funcs[key] = obs_func
        
        if self._include_objects:
            for i in range(self._num_objects):
                key = f'object_{i}'
                obs_func = self.get_observe_object(i)
                if self._flatten_objects:
                    flat_map[key] = (flat_idx, flat_idx+self._object_dim, obs_func)
                    flat_idx += self._object_dim
                else:
                    dict_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._object_dim,), dtype=float)
                    dict_obs_funcs[key] = obs_func

        if self._include_arena:
            arena_one_hot_dim = (len(self._arena_one_hot) if self._arena_one_hot is not None else 1)
            arena_key = ObservationSpecification.ARENA_KEY
            if self._flatten_arena:
                obs_func = self.get_observe_arena()
                arena_obs_size = self._arena_size[0] * self._arena_size[1] * arena_one_hot_dim
                flat_map[arena_key] = (flat_idx, flat_idx + arena_obs_size, obs_func)
                flat_idx += arena_obs_size
            else:
                obs_func = self.get_observe_arena()
                dict_obs_funcs[arena_key] = obs_func
                if self._arena_one_hot is None:
                    dict_spaces[arena_key] = gym.spaces.Box(low=0, high=4, shape=self._arena_size, dtype=int)
                else:
                    dict_spaces[arena_key] = gym.spaces.Box(low=0, high=1, shape=(*self._arena_size, arena_one_hot_dim), dtype=float)

        feature_key = ObservationSpecification.FEATURES_KEY
        scale_key = ObservationSpecification.SCALE_KEY
        if flat_idx > 0:
            if self._include_xy_scale:
                flat_map[scale_key] = (flat_idx, flat_idx+1, lambda state: state.xy_scale)
                flat_idx += 1
            dict_spaces[feature_key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_idx,), dtype=float)
            dict_obs_funcs[feature_key] = self.get_observe_flat_features(flat_map, flat_idx)
        if self._include_xy_scale:
            dict_spaces[scale_key] = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)
            dict_obs_funcs[scale_key] = lambda state: state.xy_scale

        self._flat_map = flat_map
        
        if len(dict_spaces) > 1:
            self._observation_space = gym.spaces.Dict(dict_spaces)
            self._observation_function = lambda state: { key:dict_obs_funcs[key](state) for key in dict_obs_funcs.keys()}
        elif len(dict_spaces) == 1:
            # flat features is only element
            self._observation_space = dict_spaces[feature_key]
            self._observation_function = dict_obs_funcs[feature_key]
        else:
            raise Exception("Observation space is empty")

        return self._observation_space, self._observation_function

    def get_observe_object(self, object_idx):

        def observe_object(state: MinimujoState):
            obj = state.objects[object_idx]
            obs = np.zeros(self._object_dim)
            idx = 0
            type = ObservationSpecification.get_maybe_one_hot(obj[0], self._object_type_one_hot)
            obs[idx:idx+len(type)] = type
            idx += len(type)

            obs[idx:idx+ObservationSpecification.OBJECT_POSE_DIM] = obj[1:1+ObservationSpecification.OBJECT_POSE_DIM]
            idx += ObservationSpecification.OBJECT_POSE_DIM

            color = ObservationSpecification.get_maybe_one_hot(obj[-2], self._object_color_one_hot)
            obs[idx:idx+len(color)] = color
            idx += len(color)

            state = ObservationSpecification.get_maybe_one_hot(obj[-1], self._object_state_one_hot)
            obs[idx:idx+len(state)] = state
            
            return obs
        return observe_object
    
    def get_observe_arena(self):
        if self._arena_remove_border:
            low_y, low_x = (1,1)
            high_y, high_x = (1+self._arena_size[0], 1+self._arena_size[1])
            if self._flatten_arena:
                return lambda state: ObservationSpecification.get_maybe_one_hot(state.grid[low_y:high_y, low_x:high_x], self._arena_one_hot) \
                    .flatten()
            else:
                return lambda state: ObservationSpecification.get_maybe_one_hot(state.grid[low_y:high_y, low_x:high_x], self._arena_one_hot)
        else:
            if self._flatten_arena:
                return lambda state: ObservationSpecification.get_maybe_one_hot(state.grid, self._arena_one_hot).flatten()
            else:
                return lambda state: ObservationSpecification.get_maybe_one_hot(state.grid, self._arena_one_hot)
    
    def get_observe_flat_features(self, flat_map: Dict[str, Tuple[int, int, Callable]], flat_dim: int):
        def observe_flat(state: MinimujoState):
            obs = np.zeros(flat_dim)
            for (low_idx, high_idx, obs_func) in flat_map.values():
                obs[low_idx:high_idx] = obs_func(state)
            return obs
        return observe_flat            

    @staticmethod
    def get_maybe_one_hot(value: Union[int, np.ndarray], one_hot_indices: Optional[np.ndarray]) -> np.ndarray:
        if one_hot_indices is None:
            # do not make one hot
            return np.atleast_1d(value)

        # Create a one-hot representation by comparing `value` with `one_hot_indices`
        one_hot = (np.asarray(value)[..., np.newaxis] == one_hot_indices).astype(int)
        return one_hot
    
OBSERVATION_REGISTRY: Dict[str, ObservationSpecification] = {
    'full': ObservationSpecification(),
    'no-arena': ObservationSpecification(include_arena=False)
}

def get_observation_spec(observation_type: Optional[Union[str, ObservationSpecification]]) -> ObservationSpecification:
    if isinstance(observation_type, str):
        assert observation_type in OBSERVATION_REGISTRY.keys(), f"observation_type {observation_type} not recognized"
        return OBSERVATION_REGISTRY[observation_type]
    elif isinstance(observation_type, ObservationSpecification):
        return observation_type
    else:
        return ObservationSpecification()