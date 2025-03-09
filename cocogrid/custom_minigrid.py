
from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv, MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Box, Ball
import numpy as np

from cocogrid.state.tasks import DEFAULT_TASK_REGISTRY, get_grid_goal_task, get_random_objects_task
from cocogrid.minigrid.doorkeycrossing import DoorKeyCrossingEnv

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
        num_objects = 1,
        colors = ['blue', 'red'],
        objects = ['ball', 'box'],
        goal_positions = 'all',
        obj_positions = 'all',
        max_steps: int = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        size = 5
        if max_steps is None:
            max_steps = 4 * size**2

        self.color_choices = colors
        object_map = {
            'ball': Ball,
            'box': Box
        }
        self.object_choices = [object_map[obj] for obj in objects]

        self.goal_positions = goal_positions
        if goal_positions == 'all':
            self.goal_positions = [(i, j) for i in range(1,size-1) for j in range(1,size-1) if (i,j) != (size//2, size // 2)]
        self.obj_positions = obj_positions
        if obj_positions == 'all':
            self.obj_positions = [(i, j) for i in range(1,size-1) for j in range(1,size-1) if (i,j) != (size//2, size // 2)]

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
    def _gen_mission():
        return "an empty room to play around with objects"
    
    def _gen_grid(self, width, height):

        # Create an walled empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the middle
        self.agent_pos = np.array(((width) // 2, (height) // 2))
        self.agent_dir = 0

        positions = self.obj_positions.copy()
        
        target_pos = tuple(self.np_random.choice(self.goal_positions))
        if target_pos in positions:
            positions.remove(target_pos)

        for i in range(self._num_objects):
            color = self.np_random.choice(self.color_choices)
            cls = self.np_random.choice(self.object_choices)
            pos_idx = self.np_random.integers(0, len(positions))
            pos = positions[pos_idx]
            positions.remove(pos)
            self.put_obj(cls(color), *pos)

            if i == self._num_objects-1:
                # make the last placed object the target
                self.target = (color, cls, *target_pos)
                self.put_obj(Goal(), *target_pos)

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

CUSTOM_ENVS = [UMazeEnv, RandomCornerEnv, HallwayChoiceEnv, RandomObjectsEnv]

CUSTOM_ENV_IDS = []

def register_custom_minigrid() -> list[str]:
    register(
        id='MiniGrid-UMaze-v0',
        entry_point='cocogrid.custom_minigrid:UMazeEnv'
    )
    DEFAULT_TASK_REGISTRY[UMazeEnv] = get_grid_goal_task
    register(
        id='MiniGrid-RandomCorner-v0',
        entry_point='cocogrid.custom_minigrid:RandomCornerEnv'
    )
    DEFAULT_TASK_REGISTRY[RandomCornerEnv] = get_grid_goal_task
    register(
        id='MiniGrid-HallwayChoice-v0',
        entry_point='cocogrid.custom_minigrid:HallwayChoiceEnv'
    )
    DEFAULT_TASK_REGISTRY[HallwayChoiceEnv] = get_grid_goal_task
    register(
        id='MiniGrid-Warehouse-v0',
        entry_point='cocogrid.custom_minigrid:WarehouseEnv'
    )
    DEFAULT_TASK_REGISTRY[WarehouseEnv] = get_grid_goal_task
    register(
        id='MiniGrid-RandomObjects-3-v0',
        entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
        kwargs={
            'num_objects': 3
        }
    )
    register(
        id='MiniGrid-RandomObjects-3-yellow-green-v0',
        entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
        kwargs={
            'num_objects': 3,
            'colors': ['yellow', 'green']
        }
    )
    register(
        id='MiniGrid-RandomObjects-3-goal-left-v0',
        entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
        kwargs={
            'num_objects': 3,
            'goal_positions': [(1,1), (1,2),(1,3),(2,1)],
            'obj_positions': [(3,1), (3,2), (3,3), (2,3)]
        }
    )
    register(
        id='MiniGrid-RandomObjects-3-goal-right-color-v0',
        entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
        kwargs={
            'num_objects': 3,
            'goal_positions': [(3,1), (3,2), (3,3), (2,3)],
            'obj_positions': [(1,1), (1,2),(1,3),(2,1)],
            'colors': ['yellow', 'green', 'purple', 'grey']
        }
    )
    DEFAULT_TASK_REGISTRY[RandomObjectsEnv] = get_random_objects_task
    register(
        id='MiniGrid-RandomObject-v0',
        entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
        kwargs={
            'num_objects': 1
        }
    )

    DEFAULT_TASK_REGISTRY[DoorKeyCrossingEnv] = get_grid_goal_task
    register(
        id='MiniGrid-DoorKeyCrossingS9N3-v0',
        entry_point='cocogrid.minigrid.doorkeycrossing:DoorKeyCrossingEnv',
        kwargs={
            'size': 9,
            'num_crossings': 3,
            'obstacle_type': DoorKeyCrossingEnv.WALL_TYPE
        }
    )

    register(
        id='MiniGrid-DoorKey-10x10-v0',
        entry_point='minigrid.envs.doorkey:DoorKeyEnv',
        kwargs={
            'size': 10
        }
    )

    register(
        id='MiniGrid-DoorKey-12x12-v0',
        entry_point='minigrid.envs.doorkey:DoorKeyEnv',
        kwargs={
            'size': 12
        }
    )

    return []

default_tasks = {
    RandomObjectsEnv: get_random_objects_task
}