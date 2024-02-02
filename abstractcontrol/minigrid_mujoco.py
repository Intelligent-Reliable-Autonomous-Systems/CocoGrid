from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.tasks.random_goal_maze import DEFAULT_ALIVE_THRESHOLD, DEFAULT_CONTROL_TIMESTEP, DEFAULT_PHYSICS_TIMESTEP
from dm_control.locomotion.props import target_sphere
from dm_control import mjcf

from abstractcontrol.door_entity import DoorEntity

class MinigridMujoco(random_goal_maze.RepeatSingleGoalMaze):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

        # self._target2 = target_sphere.TargetSphere(radius=3)
        self._target2 = DoorEntity()
        self._target2pose = (6.5, 6.5, 0)
        self._maze_arena.attach(self._target2)

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        self._target_position = self._maze_arena.target_positions[
            random_state.randint(0, len(self._maze_arena.target_positions))]
        mjcf.get_attachment_frame(
            self._target2.mjcf_model).pos = self._target2pose