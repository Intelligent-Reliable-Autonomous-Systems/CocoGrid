import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Door, Key, Wall, Ball, Box
from minigrid.minigrid_env import MiniGridEnv

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
