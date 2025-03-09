import collections
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.walkers import base
from minigrid.minigrid_env import MiniGridEnv

from cocogrid.state.cocogrid_state import CocogridState

if TYPE_CHECKING:
    from cocogrid.mujoco.cocogrid_arena import CocogridArena
    from cocogrid.mujoco.mujoco_agent import MuJoCoAgent

OBJECT_SIZE = 16


class CocogridStateObserver:

    def __init__(self, minigrid_env: MiniGridEnv, xy_scale: float, objects, agent: "MuJoCoAgent"):
        self.grid = CocogridState.get_grid_state_from_minigrid(minigrid_env)
        self.xy_scale = xy_scale
        self.objects = objects
        self.agent = agent
        self.walker_observables = collections.OrderedDict({})
        for observable in (agent.walker.observables.proprioception +
                           agent.walker.observables.kinematic_sensors +
                            agent.walker.observables.dynamic_sensors):
            observable.enabled = True
        self.walker_observables.update({k:v for k,v in agent.walker.observables.as_dict().items() if v.enabled})

    def get_state(self, physics):
        """Get the grid, objects, pose, and walker states for the current physics state."""
        object_states = np.array([obj.get_object_state(physics) for obj in self.objects])
        walker_pose = np.zeros(13)
        walker_pose[:3] = physics.bind(self.agent.walker.root_body).xpos
        walker_pose[3:7] = physics.bind(self.agent.walker.root_body).xquat
        walker_pose[7:13] = physics.bind(self.agent.walker.root_body).cvel
        walker = { key:observable.observation_callable(physics)().copy() for key, observable in self.walker_observables.items()}

        return CocogridState(self.grid, self.xy_scale, object_states, walker_pose, walker)

class ObservableSpecification(ABC):
    @staticmethod
    def get_walker_pos_vel_observables(arena, walker):
        freejoints = [joint for joint in arena.mjcf_model.find_all('joint') if joint.tag == 'freejoint']
        if len(freejoints) > 0:
            def get_walker_pos(physics):
                walker_pos = physics.bind(freejoints[0]).qpos
                return walker_pos
            def get_walker_vel(physics):
                    walker_vel = physics.bind(freejoints[0]).qvel
                    return walker_vel
        else:
            def get_walker_pos(physics):
                walker_pos = physics.bind(walker.root_body).xpos
                return walker_pos
            def get_walker_vel(physics):
                    walker_vel = physics.bind(walker.root_joints).qvel
                    return walker_vel[:2]
        
        return observable_lib.Generic(get_walker_pos), observable_lib.Generic(get_walker_vel)
    
    @staticmethod
    def get_object_observable(arena: "CocogridArena", num_objects: int):
        target_size = num_objects * OBJECT_SIZE
        def get_objects(physics):
            if arena.current_state is None:
                return np.zeros(target_size)
            flat = arena.current_state.objects.flatten()[:target_size]
            return np.pad(flat, target_size - len(flat))
        return observable_lib.Generic(get_objects)

class FullVectorSpecification(ObservableSpecification):

    def __init__(self, state_observer: CocogridStateObserver, arena: "CocogridArena"):
        self._state_observer = state_observer
        self._arena = arena
        self._observables = collections.OrderedDict({})

        # agent
        self._observables['agent_pos'] = observable_lib.Generic(arena._agent.get_walker_pos)
        self._observables['agent_vel'] = observable_lib.Generic(arena._agent.get_walker_vel)

        # objects
        self._num_objects = len(state_observer.objects)
        if self._num_objects > 0:
            obj_obs = ObservableSpecification.get_object_observable(arena, self._num_objects)
            self._observables['objects'] = obj_obs

        # walker
        for key, walker_observable in state_observer.walker_observables.items():
            self._observables[key] = walker_observable

    def enable(self):
        for observable in self._observables.values():
            observable.enabled = True

    @property
    def observables(self):
        return self._observables