import collections
from dm_control import composer
from dm_control.composer.observation import observable as observable_lib
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

_NUM_RAYS = 10

# Aliveness in [-1., 0.].
DEFAULT_ALIVE_THRESHOLD = -0.5

DEFAULT_PHYSICS_TIMESTEP = 0.001
DEFAULT_CONTROL_TIMESTEP = 0.025

class MinimujoTask(composer.Task):
    """A base task for minimujo arenas."""

    def __init__(self,
               walker,
               minimujo_arena,
               observation_type='top_camera',
               reward_type='sparse',
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=False,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
        """Initializes goal-directed maze task.
        Args:
        walker: The body to navigate the maze.
        minimujo_arena: The physical maze arena object.
        randomize_spawn_position: Flag to randomize position of spawning.
        randomize_spawn_rotation: Flag to randomize orientation of spawning.
        rotation_bias_factor: A non-negative number that concentrates initial
            orientation away from walls. When set to zero, the initial orientation
            is uniformly random. The larger the value of this number, the more
            likely it is that the initial orientation would face the direction that
            is farthest away from a wall.
        aliveness_reward: Reward for being alive.
        aliveness_threshold: Threshold if should terminate based on walker
            aliveness feature.
        contact_termination: whether to terminate if a non-foot geom touches the
            ground.
        physics_timestep: timestep of simulation.
        control_timestep: timestep at which agent changes action.
        """

        self._walker = walker
        self._minimujo_arena = minimujo_arena
        self._walker.create_root_joints(self._minimujo_arena.attach(self._walker))
        self._minimujo_arena.register_walker(self._walker)

        self._randomize_spawn_position = randomize_spawn_position
        self._randomize_spawn_rotation = randomize_spawn_rotation
        self._rotation_bias_factor = rotation_bias_factor

        self._aliveness_reward = aliveness_reward
        self._aliveness_threshold = aliveness_threshold
        self._contact_termination = contact_termination
        self._discount = 1.0

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)
        
        self.observation_types = observation_type.split(',')

        self._task_observables = collections.OrderedDict({})
        if 'top_camera' in self.observation_types:
            self._minimujo_arena.observables.get_observable('top_camera').enabled = True
            self._task_observables.update({'top_camera': self._minimujo_arena.observables.get_observable('top_camera')})
        if 'pos' in self.observation_types:
            def get_walker_pos(physics):
                walker_pos = physics.bind(self._walker.root_body).xpos
                return walker_pos[:2]
            absolute_position = observable_lib.Generic(get_walker_pos)
            absolute_position.enabled = True

            self._task_observables.update({'abs_pos': absolute_position})
        if 'walker' in self.observation_types:
            for observable in (self._walker.observables.proprioception +
                            self._walker.observables.kinematic_sensors +
                            self._walker.observables.dynamic_sensors):
                observable.enabled = True
        if 'egocentric_camera' in self.observation_types:
            # self._walker.observables.egocentric_camera.height = 64
            # self._walker.observables.egocentric_camera.width = 64
            self._walker.observables.egocentric_camera.enabled = True
        walker_observables = {k:v for k,v in self._walker.observables.as_dict().items() if v.enabled}
        self._task_observables.update(walker_observables)

        self._cum_reward = 0

        reward_func_map = {
            'sparse': self._sparse_reward,
            'sparse_cost': self._sparse_cost,
            'subgoal': self._subgoal_reward,
            'subgoal_cost': self._subgoal_cost,
            'subgoal_dense': self._subgoal_reward # the dense flag has been set in arena
        }
        self.reward_type = reward_type
        if not reward_type in reward_func_map:
            raise Exception(f'reward_type {reward_type} must be one of {list(reward_func_map.keys())}')
        self._reward_func = reward_func_map[reward_type]

    @property
    def observables(self):
        return self._task_observables
    
    @property
    def task_observables(self):
        return self._task_observables

    @property
    def name(self):
        return 'minimujo'

    @property
    def root_entity(self):
        return self._minimujo_arena

    def initialize_episode_mjcf(self, unused_random_state):
        self._minimujo_arena.regenerate()

    def _respawn(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        if self._randomize_spawn_position:
            self._spawn_position = self._minimujo_arena.spawn_positions[
                random_state.randint(0, len(self._minimujo_arena.spawn_positions))]
        else:
            self._spawn_position = self._minimujo_arena.spawn_positions[0]

        if self._randomize_spawn_rotation:
            # Move walker up out of the way before raycasting.
            self._walker.shift_pose(physics, [0.0, 0.0, 100.0])

            distances = []
            geomid_out = np.array([-1], dtype=np.intc)
            for i in range(_NUM_RAYS):
                theta = 2 * np.pi * i / _NUM_RAYS
                pos = np.array([self._spawn_position[0], self._spawn_position[1], 0.1],
                            dtype=np.float64)
                vec = np.array([np.cos(theta), np.sin(theta), 0], dtype=np.float64)
                dist = mjbindings.mjlib.mj_ray(
                    physics.model.ptr, physics.data.ptr, pos, vec,
                    None, 1, -1, geomid_out)
                distances.append(dist)

            def remap_with_bias(x):
                """Remaps values [-1, 1] -> [-1, 1] with bias."""
                return np.tanh((1 + self._rotation_bias_factor) * np.arctanh(x))

            max_theta = 2 * np.pi * np.argmax(distances) / _NUM_RAYS
            rotation = max_theta + np.pi * (
                1 + remap_with_bias(random_state.uniform(-1, 1)))

            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

            # Move walker back down.
            self._walker.shift_pose(physics, [0.0, 0.0, -100.0])
        else:
            quat = None

        self._walker.shift_pose(
            physics, [self._spawn_position[0], self._spawn_position[1], 0.0],
            quat,
            rotate_velocity=True)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._minimujo_arena.initialize_arena(physics, random_state)
        self._respawn(physics, random_state)
        self._discount = 1.0

        walker_foot_geoms = set(self._walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms]
        self._walker_nonfoot_geomids = set(
            physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(
            physics.bind(self._minimujo_arena.ground_geoms).element_id)
        
        self._cum_reward = 0
        self._subgoal_reward_decay = 0.05
        self._steps_since_last_subgoal = 0

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break
        self._steps_since_last_subgoal += 1

    def should_terminate_episode(self, physics):
        # if self._walker.aliveness(physics) < self._aliveness_threshold:
        #     self._failure_termination = True
        if self._minimujo_arena._terminated:
            self._discount = 0.0
            return True
        else:
            return False

    def get_reward(self, physics):
        del physics
        reward = self._reward_func(self._minimujo_arena._extrinsic_reward, self._minimujo_arena._intrinsic_reward)
        self._cum_reward += reward

        return reward

    def get_discount(self, physics):
        del physics
        return self._discount
    
    def _sparse_reward(self, extrinsic, intrinsic):
        return extrinsic
        
    def _sparse_cost(self, extrinsic, intrinsic):
        return -1 * (extrinsic <= 0)

    def _subgoal_reward(self, extrinsic, intrinsic):
        return extrinsic + intrinsic
    
    def _subgoal_cost(self, extrinsic, intrinsic):
        if extrinsic > 0 or intrinsic > 0:
            self._steps_since_last_subgoal = 0
        return extrinsic + np.exp(-self._subgoal_reward_decay * self._steps_since_last_subgoal) - 1