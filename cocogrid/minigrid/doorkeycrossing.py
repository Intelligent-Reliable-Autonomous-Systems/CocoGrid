# Adapted from https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/crossing.py

from __future__ import annotations

import itertools as itt

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class DoorKeyCrossingEnv(MiniGridEnv):

    WALL_TYPE = Wall
    LAVA_TYPE = Lava

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.goal_position = None

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
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
    def _gen_mission_lava():
        return "avoid the lava, unlock the door, and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "unlock the door and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_position = (width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        hole_positions = []
        for idx, direction in enumerate(path):
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            hole_positions.append((i,j))
            self.grid.set(i, j, None)
        door_key_pos_found = False
        tries_left = 100 # it should hardly ever take more than 3
        while not door_key_pos_found:
            door_idx = self.np_random.integers(0, len(path))
            door_pos = hole_positions[door_idx]

            reachable = self._get_reachable_cells((1,1), door_pos)
            if len(reachable) == 0:
                # none reachable; sample another door position
                continue
            # key_pos = self.np_random.integers((1,1), (width - 1, height - 1))
            key_pos = self.np_random.choice(reachable)
            
            self.grid.set(*door_pos, Door('yellow', is_locked=True))
            self.grid.set(*key_pos, Key('yellow'))
            door_key_pos_found = True

            tries_left -= 1
            if tries_left == 0:
                # the generated grid cannot place the key. resample the walls
                self._gen_grid(width, height)
                return
            

    def _get_reachable_cells(self, start, door_pos):
        from collections import deque

        # Offsets for the 4 directions (up, right, down, left)
        DIRS = [(-1,0), (0,1), (1,0), (0,-1)]
        
        (sx, sy) = start
        visited = set()
        visited.add((sx, sy))
        queue = deque()
        queue.append((sx, sy))

        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS:
                nx, ny = x+dx, y+dy
                # Check grid bounds
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                # Check if it's traversable (i.e. not a wall/door/lava, etc.)
                cell = self.grid.get(nx, ny)
                if cell is not None or (nx, ny) == door_pos:
                    continue
                # Not visited yet?
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        visited.remove(start)
        return list(visited)