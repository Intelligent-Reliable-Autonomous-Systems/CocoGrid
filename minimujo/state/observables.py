

from abc import ABC
import collections

from dm_control.composer.observation import observable as observable_lib
import numpy as np

from minimujo.minimujo_arena import MinimujoArena
from minimujo.state.minimujo_state import MinimujoStateObserver

OBJECT_SIZE = 16

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
    def get_object_observable(arena: MinimujoArena, num_objects: int):
        target_size = num_objects * OBJECT_SIZE
        def get_objects(physics):
            if arena.current_state is None:
                return np.zeros(target_size)
            flat = arena.current_state.objects.flatten()[:target_size]
            return np.pad(flat, target_size - len(flat))
        return observable_lib.Generic(get_objects)

class FullVectorSpecification(ObservableSpecification):

    def __init__(self, state_observer: MinimujoStateObserver, arena: MinimujoArena):
        self._state_observer = state_observer
        self._arena = arena
        self._observables = collections.OrderedDict({})

        # agent
        pos_obs, vel_obs = ObservableSpecification.get_walker_pos_vel_observables(arena, arena._walker)
        self._observables['agent_pos'] = pos_obs
        self._observables['agent_vel'] = vel_obs

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