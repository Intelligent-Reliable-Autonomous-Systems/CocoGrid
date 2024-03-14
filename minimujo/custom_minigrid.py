
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv, MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
import numpy as np

class UMazeEnv(MiniGridEnv):

    def __init__(
        self,
        size=5,
        max_steps: int | None = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "navigate the U-bend and get to the green goal square"
    
    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        mid_height = height // 2
        self.grid.wall_rect(0, mid_height, width - 2, 1)

        # Place the agent in the bottom-left corner
        self.agent_pos = np.array((1, height-2))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 1, 1)

CUSTOM_ENVS = [UMazeEnv]

def register_custom_minigrid():
    register(
        id='MiniGrid-UMaze-v0',
        entry_point='minimujo.custom_minigrid:UMazeEnv'
    )