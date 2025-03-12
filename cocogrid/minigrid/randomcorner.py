from __future__ import annotations

from typing import Any

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv, MissionSpace


class RandomCornerEnv(MiniGridEnv):
    """A Minigrid environment with an empty room and an goal placed in a random corner."""

    def __init__(
        self,
        size: int = 7,
        max_steps: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Construct a RandomCornerEnv."""
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Get to the green goal."

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the middle
        self.agent_pos = np.array(((width) // 2, (height) // 2))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        corner = self.np_random.choice(4)
        if corner == 0:
            self.put_obj(Goal(), 1, 1)
        elif corner == 1:
            self.put_obj(Goal(), 1, height - 2)
        elif corner == 2:
            self.put_obj(Goal(), width - 2, 1)
        else:
            self.put_obj(Goal(), width - 2, height - 2)
