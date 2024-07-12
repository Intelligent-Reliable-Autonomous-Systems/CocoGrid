import collections
from dataclasses import dataclass
from dm_control.locomotion.walkers import base
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

@dataclass
class MinimujoState:

    def __init__(self, grid, xy_scale, objects, pose, walker):
        self.grid = grid
        self.xy_scale = xy_scale
        self.objects = objects
        self.pose = pose
        self.walker = walker

    def get_walker_position(self):
        return self.pose[:3]
    
    def get_normalized_walker_position(self):
        pos = self.pose[:3].copy()
        pos[:2] /= self.xy_scale * self.grid.shape[0]
        pos[1] *= -1
        return pos

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
        walker = { key:observable.observation_callable(physics)() for key, observable in self.walker_observables.items()}

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