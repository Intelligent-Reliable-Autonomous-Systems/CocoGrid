
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv, MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Box, Ball
import numpy as np

from minimujo.state.tasks import get_random_objects_task

class UMazeEnv(MiniGridEnv):

    def __init__(
        self,
        size=5,
        max_steps: int = None,
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

class RandomCornerEnv(MiniGridEnv):
    def __init__(
        self,
        size=7,
        max_steps: int = None,
        **kwargs,
    ):

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
    def _gen_mission():
        return "navigate an empty room to a goal in a random corner"
    
    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

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

class RandomObjectsEnv(MiniGridEnv):
    def __init__(
        self,
        max_steps: int = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        size = 5
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
    def _gen_mission():
        return "an empty room to play around with objects"
    
    def _gen_grid(self, width, height):

        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the middle
        self.agent_pos = np.array(((width) // 2, (height) // 2))
        self.agent_dir = 0

        positions = [(i, j) for i in range(1,width-1) for j in range(1,height-1)]
        positions.remove((width//2, height // 2))
        num_objects = 3
        for i in range(num_objects):
            color = np.random.choice(['blue', 'red'])
            cls = np.random.choice([Ball, Box])
            pos_idx = np.random.randint(len(positions))
            pos = positions[pos_idx]
            positions.remove(pos)
            self.put_obj(cls(color), *pos)

            if i == num_objects-1:
                target_pos_idx = np.random.randint(len(positions))
                target_pos = positions[target_pos_idx]
                positions.remove(target_pos)
                self.target = (color, cls, *target_pos)

    # def step(action):
    #     obs, _, _, trunc, info = super().step(action)



class HallwayChoiceEnv(MiniGridEnv):
    def __init__(
        self,
        size=7,
        max_steps: int = None,
        **kwargs,
    ):

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
    def _gen_mission():
        return "navigate a hallway and choose which path to go down"
    
    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Create hallways
        self.grid.wall_rect(0, 0, width - 2, height // 2)
        self.grid.wall_rect(0, height // 2 + 1, width - 2, height // 2)

        # Place the agent in the middle
        self.agent_pos = np.array((1, (height) // 2))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        corner = self.np_random.choice(2)
        if corner == 0:
            self.put_obj(Goal(), width - 2, 1)
        else:
            self.put_obj(Goal(), width - 2, height - 2)

class WarehouseEnv(MiniGridEnv):
    def __init__(
        self,
        max_steps: int = None,
        **kwargs,
    ):
        size = 17

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
    def _gen_mission():
        return "navigate a warehouse with boxes"
    
    def _gen_grid(self, width, height):

        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the middle
        self.agent_pos = np.array((12, 12))
        self.agent_dir = 0

        self.grid.vert_wall(3, 2, 9)
        self.grid.horz_wall(2, 2, 3)
        self.grid.horz_wall(2, 6, 3)
        self.grid.horz_wall(2, 10, 3)

        self.grid.vert_wall(8, 2, 9)
        self.grid.horz_wall(7, 2, 3)
        self.grid.horz_wall(7, 6, 3)
        self.grid.horz_wall(7, 10, 3)

        self.grid.vert_wall(13, 2, 9)
        self.grid.horz_wall(12, 2, 3)
        self.grid.horz_wall(12, 6, 3)
        self.grid.horz_wall(12, 10, 3)

        self.grid.horz_wall(3, 13, 9)
        self.grid.vert_wall(3, 13, 2)
        self.grid.vert_wall(11, 13, 2)

        self.put_obj(Box(color='red'), 4, 4)
        self.put_obj(Box(color='blue'), 7, 9)
        self.put_obj(Box(color='yellow'), 6, 14)
        self.put_obj(Box(color='purple'), 14, 7)

        self.put_obj(Goal(), 14, 14)
        self.put_obj(Goal(), 14, 15)
        self.put_obj(Goal(), 15, 14)
        self.put_obj(Goal(), 15, 15)

CUSTOM_ENVS = [UMazeEnv, RandomCornerEnv, HallwayChoiceEnv]

def register_custom_minigrid():
    register(
        id='MiniGrid-UMaze-v0',
        entry_point='minimujo.custom_minigrid:UMazeEnv'
    )
    register(
        id='MiniGrid-RandomCorner-v0',
        entry_point='minimujo.custom_minigrid:RandomCornerEnv'
    )
    register(
        id='MiniGrid-HallwayChoice-v0',
        entry_point='minimujo.custom_minigrid:HallwayChoiceEnv'
    )
    register(
        id='MiniGrid-Warehouse-v0',
        entry_point='minimujo.custom_minigrid:WarehouseEnv'
    )
    register(
        id='MiniGrid-RandomObjects-v0',
        entry_point='minimujo.custom_minigrid:RandomObjectsEnv'
    )

default_tasks = {
    RandomObjectsEnv: get_random_objects_task
}