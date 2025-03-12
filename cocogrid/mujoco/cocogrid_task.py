from typing import TYPE_CHECKING
import numpy as np
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings

from cocogrid.mujoco.observables import FullVectorSpecification
from cocogrid.tasks import get_grid_goal_task

if TYPE_CHECKING:
    from cocogrid.mujoco.mujoco_agent import MuJoCoAgent

_NUM_RAYS = 10

# Aliveness in [-1., 0.].
DEFAULT_ALIVE_THRESHOLD = -0.5

DEFAULT_PHYSICS_TIMESTEP = 0.001
DEFAULT_CONTROL_TIMESTEP = 0.025


class CocogridTask(composer.Task):
    """A base task for cocogrid arenas."""

    def __init__(
        self,
        agent: "MuJoCoAgent",
        cocogrid_arena,
        observation_type="pos,vel,walker",
        get_task_function=get_grid_goal_task,
        random_rotation=False,
        rotation_bias_factor=0,
        aliveness_reward=0.0,
        aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
        contact_termination=False,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
    ):
        """Initializes goal-directed maze task.
        Args:
        walker: The body to navigate the maze.
        cocogrid_arena: The physical maze arena object.
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

        self._agent = agent
        self._cocogrid_arena = cocogrid_arena
        self._agent.walker.create_root_joints(self._cocogrid_arena.attach(self._agent.walker))
        self._cocogrid_arena.register_agent(agent)

        self._randomize_spawn_rotation = random_rotation
        self._rotation_bias_factor = rotation_bias_factor

        self._aliveness_reward = aliveness_reward
        self._aliveness_threshold = aliveness_threshold
        self._contact_termination = contact_termination
        self._discount = 1.0

        self.get_task_function = get_task_function

        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)

        self.observable_spec = FullVectorSpecification(self._cocogrid_arena.get_state_observer(), self._cocogrid_arena)
        self.observable_spec.enable()

    @property
    def observables(self):
        return self.observable_spec.observables

    @property
    def task_observables(self):
        return self.observable_spec.observables

    @property
    def name(self):
        return "cocogrid"

    @property
    def root_entity(self):
        return self._cocogrid_arena

    def initialize_episode_mjcf(self, random_state):
        self._cocogrid_arena.initialize_arena_mjcf(random_state)

    def _respawn(self, physics, random_state):
        self._agent.walker.reinitialize_pose(physics, random_state)

        self._spawn_position = self._cocogrid_arena.spawn_positions[0]

        if self._randomize_spawn_rotation:
            # Move walker up out of the way before raycasting.
            self._agent.walker.shift_pose(physics, [0.0, 0.0, 100.0])

            distances = []
            geomid_out = np.array([-1], dtype=np.intc)
            for i in range(_NUM_RAYS):
                theta = 2 * np.pi * i / _NUM_RAYS
                pos = np.array(
                    [self._spawn_position[0], self._spawn_position[1], 0.1],
                    dtype=np.float64,
                )
                vec = np.array([np.cos(theta), np.sin(theta), 0], dtype=np.float64)
                dist = mjbindings.mjlib.mj_ray(
                    physics.model.ptr,
                    physics.data.ptr,
                    pos,
                    vec,
                    None,
                    1,
                    -1,
                    geomid_out,
                )
                distances.append(dist)

            def remap_with_bias(x):
                """Remaps values [-1, 1] -> [-1, 1] with bias."""
                return np.tanh((1 + self._rotation_bias_factor) * np.arctanh(x))

            max_theta = 2 * np.pi * np.argmax(distances) / _NUM_RAYS
            rotation = max_theta + np.pi * (1 + remap_with_bias(random_state.uniform(-1, 1)))

            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

            # Move walker back down.
            self._agent.walker.shift_pose(physics, [0.0, 0.0, -100.0])
        else:
            quat = None

        self._agent.walker.shift_pose(
            physics,
            [self._spawn_position[0], self._spawn_position[1], 0.0],
            quat,
            rotate_velocity=True,
        )

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._cocogrid_arena.initialize_arena(physics, random_state)
        self._respawn(physics, random_state)

        walker_foot_geoms = set(self._agent.walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._agent.walker.mjcf_model.find_all("geom") if geom not in walker_foot_geoms
        ]
        self._agent.walker_nonfoot_geomids = set(physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(physics.bind(self._cocogrid_arena.ground_geoms).element_id)

        self._discount = 1.0
        self._reward = 0
        self._cum_reward = 0
        self._termination = False

        self._subgoal_reward_decay = 0.05
        self._steps_since_last_subgoal = 0
        self._task_function, self._task_description = self.get_task_function(self._cocogrid_arena._minigrid)

        self._cocogrid_arena.current_state = self._cocogrid_arena.state_observer.get_state(physics)

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._agent.walker_nonfoot_geomids, self._ground_geomids
        return (contact.geom1 in set1 and contact.geom2 in set2) or (contact.geom1 in set2 and contact.geom2 in set1)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break

        self._cocogrid_arena.update_state(physics)
        self._reward, self._termination = self._task_function(
            self._cocogrid_arena.previous_state, self._cocogrid_arena.current_state
        )
        self._cum_reward += self._reward

    def should_terminate_episode(self, physics):
        # if self._walker.aliveness(physics) < self._aliveness_threshold:
        #     self._failure_termination = True
        if self._termination:
            self._discount = 0.0
            return True
        return False

    def get_reward(self, physics):
        del physics
        return self._reward

    def get_discount(self, physics):
        del physics
        return self._discount

    @property
    def reward_total(self):
        return self._cum_reward

    @property
    def terminated(self):
        return self._termination

    @property
    def function(self):
        return self._task_function

    @property
    def description(self):
        return self._task_description
