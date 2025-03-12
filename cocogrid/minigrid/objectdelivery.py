import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from cocogrid.common.abstraction.grid_abstraction import GridAbstraction
from cocogrid.common.cocogrid_state import CocogridState
from cocogrid.common.entity import ObjectEnum, get_color_id
from cocogrid.tasks import TaskEvalType


class ObjectDeliveryEnv(MiniGridEnv):
    """A Minigrid environment with an empty room populated by objects and a goal."""

    def __init__(
        self,
        num_objects=1,
        colors=["blue", "red"],
        objects=["ball", "box"],
        goal_positions="all",
        obj_positions="all",
        max_steps: int = None,
        **kwargs,
    ):
        size = 5
        if max_steps is None:
            max_steps = 4 * size**2

        self.color_choices = colors
        object_map = {"ball": Ball, "box": Box}
        self.object_choices = [object_map[obj] for obj in objects]

        self.goal_positions = goal_positions
        if goal_positions == "all":
            self.goal_positions = [
                (i, j) for i in range(1, size - 1) for j in range(1, size - 1) if (i, j) != (size // 2, size // 2)
            ]
        self.obj_positions = obj_positions
        if obj_positions == "all":
            self.obj_positions = [
                (i, j) for i in range(1, size - 1) for j in range(1, size - 1) if (i, j) != (size // 2, size // 2)
            ]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[colors, objects, self.goal_positions],
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )
        self._num_objects = num_objects
        assert 1 <= num_objects <= 8

    @staticmethod
    def _gen_mission(color: str, object: str, goal_pos: tuple[int, int]) -> str:
        return f"Deliver a {color} {object} to tile ({goal_pos[0]}, {goal_pos[1]})."

    def _gen_grid(self, width, height):
        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the middle
        self.agent_pos = np.array(((width) // 2, (height) // 2))
        self.agent_dir = 0

        positions = self.obj_positions.copy()

        goal_pos = tuple(self.np_random.choice(self.goal_positions))
        if goal_pos in positions:
            positions.remove(goal_pos)

        for i in range(self._num_objects):
            color = self.np_random.choice(self.color_choices)
            cls = self.np_random.choice(self.object_choices)
            pos_idx = self.np_random.integers(0, len(positions))
            pos = positions[pos_idx]
            positions.remove(pos)
            self.put_obj(cls(color), *pos)

            if i == self._num_objects - 1:
                # make the last placed object the target to deliver
                self.target = (color, cls, *goal_pos)
                self.put_obj(Goal(), *goal_pos)
                self.mission = f"Deliver a {color} {cls.__name__} to tile ({goal_pos[0]}, {goal_pos[1]})."

    @staticmethod
    def get_cocogrid_task(random_objects_env: "ObjectDeliveryEnv") -> tuple[TaskEvalType, str]:
        """Get the cocogrid task for the ObjectDeliveryEnv."""
        color, cls, x, y = random_objects_env.target
        color_idx = get_color_id(color)
        cls_idx = ObjectEnum.get_id(cls)
        target_object = (cls_idx, x, y, color_idx, 0)

        def object_to_position_task(prev_state: CocogridState, cur_state: CocogridState) -> tuple[float, bool]:
            """Evaluate when the object has been delivered to the goal and is stationary."""
            grid_state = GridAbstraction.from_cocogrid_state(cur_state)
            for idx, obj in enumerate(grid_state.objects):
                if obj == target_object:
                    cvel = cur_state.objects[idx, 8:11]
                    vel = np.linalg.norm(cvel)
                    if vel < 0.1:
                        return 1, True
            return 0, False

        return object_to_position_task, random_objects_env.mission
