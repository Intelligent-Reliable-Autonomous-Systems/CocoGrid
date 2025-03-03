import collections
from dataclasses import dataclass
from typing import Dict, ClassVar
from dm_control.locomotion.walkers import base
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from minimujo.color import get_color_idx
import numpy as np

@dataclass
class MinimujoState:

    POSE_IDX_POS: ClassVar[int] = 0
    POSE_IDX_QUAT: ClassVar[int] = 3
    POSE_IDX_VEL: ClassVar[int] = 7
    OBJECT_IDX_TYPE: ClassVar[int] = 0
    OBJECT_IDX_POS: ClassVar[int] = 1
    OBJECT_IDX_DOOR_ORIENTATION: ClassVar[int] = 5
    OBJECT_IDX_VEL: ClassVar[int] = 8
    OBJECT_IDX_COLOR: ClassVar[int] = 14
    OBJECT_IDX_STATE: ClassVar[int] = 15

    @staticmethod
    def color_to_idx(color):
        return get_color_idx(color)

    def __init__(self, grid: np.ndarray, xy_scale: float, objects: np.ndarray, pose: np.ndarray, walker: Dict[str, np.ndarray]):
        self.grid = grid
        self.grid.flags.writeable = False
        self.xy_scale = xy_scale
        self.objects = objects
        self.objects.flags.writeable = False
        self.pose = pose
        self.pose.flags.writeable = False
        self.walker = walker
        for arr in self.walker.values():
            if arr.ndim > 0:
                arr.flags.writeable = False

    def get_arena_size(self):
        return np.array(self.grid.shape) * self.xy_scale

    def get_walker_position(self):
        return self.pose[:3]
    
    def get_normalized_walker_position(self, without_border=False):
        pos = self.pose[:3].copy()
        pos[1] *= -1
        pos[:2] = (pos[:2] / self.xy_scale - int(without_border)) / (self.grid.shape[0] - 2 * int(without_border))
        return pos
    
    def get_walker_velocity(self):
        return self.pose[7:10]
    
    @staticmethod
    def get_object_pos_slice(dim: int = 3):
        assert 1 <= dim <= 3
        return slice(MinimujoState.OBJECT_IDX_POS, MinimujoState.OBJECT_IDX_POS+dim)

class MinimujoStateObserver:

    def __init__(self, minigrid_env: MiniGridEnv, xy_scale: float, objects, walker: base.Walker):
        self.grid = MinimujoStateObserver.get_grid_state_from_minigrid(minigrid_env)
        self.xy_scale = xy_scale
        self.objects = objects
        self.walker = walker
        self.walker_observables = collections.OrderedDict({})
        for observable in (walker.observables.proprioception +
                           walker.observables.kinematic_sensors +
                            walker.observables.dynamic_sensors):
            observable.enabled = True
        self.walker_observables.update({k:v for k,v in walker.observables.as_dict().items() if v.enabled})

    def get_state(self, physics):
        """
        Get the grid, objects, pose, and walker states for the current physics state
        """
        object_states = np.array([obj.get_object_state(physics) for obj in self.objects])
        walker_pose = np.zeros(13)
        walker_pose[:3] = physics.bind(self.walker.root_body).xpos
        walker_pose[3:7] = physics.bind(self.walker.root_body).xquat
        walker_pose[7:13] = physics.bind(self.walker.root_body).cvel
        walker = { key:observable.observation_callable(physics)().copy() for key, observable in self.walker_observables.items()}

        return MinimujoState(self.grid, self.xy_scale, object_states, walker_pose, walker)


    @staticmethod
    def get_grid_state_from_minigrid(minigrid_env: MiniGridEnv) -> np.ndarray:
        """
        Takes a minigrid environment and converts the grid into a static state representation.
        
        Input:
        - minigrid_env (MiniGridEnv): a minigrid environment

        Output:
        - grid_state (np.ndarray): a matrix with same dimensions as minigrid_env.grid,
            where 0 is nothing, 1 is wall, 2 is goal, and 3 is lava
        """
        object_mapping = {
            Wall: 1,
            Goal: 2,
            Lava: 3
        }
        width, height = minigrid_env.grid.width, minigrid_env.grid.height
        grid_state = np.zeros(shape=(width, height), dtype=int)
        for i in range(width):
            for j in range(height):
                # i is the col, j is the row
                world_obj = minigrid_env.grid.get(i, j)
                grid_state[i,j] = object_mapping.get(type(world_obj), 0)

        return grid_state