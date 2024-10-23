from __future__ import annotations
from collections import deque
import hashlib
import pickle
import re
import sys
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from minimujo.color import COLOR_MAP, get_color_idx
from minimujo.state.goal_wrapper import DeterministicValueIterationPlanner, DjikstraBackwardsPlanner, GoalObserver, GoalWrapper
from minimujo.state.minimujo_state import MinimujoState

class GridAbstraction:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    ACTION_GRAB = 4
    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_GRAB]
    OBJECT_IDS = ['ball', 'box', 'door', 'key']
    DOOR_IDX = 2
    GRID_EMPTY = 0
    GRID_WALL = 1
    GRID_GOAL = 2
    GRID_LAVA = 3
    DOOR_STATE_CLOSED = 0
    DOOR_STATE_OPEN = 1

    def __init__(self, grid: np.ndarray, walker_pos: Tuple[int], objects: List[Tuple[int]]) -> None:
        self.grid: np.ndarray = grid
        self._grid_hash: str = hash_ndarray(self.grid)

        self.walker_pos = walker_pos
        
        self.objects = objects.copy()
        self._doors = [obj for obj in self.objects if obj[0] == GridAbstraction.DOOR_IDX]
        self._held_object = next((i for i, o in enumerate(self.objects) if o[0] != GridAbstraction.DOOR_IDX and o[4] > 0), -1)
        if self._held_object > -1:
            # held object should be snapped to walker position
            idx, _, _, color, state = self.objects[self._held_object]
            self.objects[self._held_object] = (idx, walker_pos[0], walker_pos[1], color, state)
        self._objects_hash = hash_list_of_tuples(self.objects)

        # solver = GridSolver()
        # solver.solve_state(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridAbstraction):
            return False
        return self.walker_pos == other.walker_pos and \
            self._grid_hash == other._grid_hash and \
            self._objects_hash == other._objects_hash
    
    def __hash__(self) -> int:
        return hash((*self.walker_pos, self._grid_hash, self._objects_hash))
    
    def __repr__(self) -> str:
        obj_str = ','.join(map(GridAbstraction.pretty_object, self.objects))
        return f'Grid[{self.walker_pos}; {obj_str}]'
    
    def do_action(self, action: int) -> GridAbstraction:
        if action < GridAbstraction.ACTION_GRAB:
            offsets = [(0,-1),(-1,0),(0,1),(1,0)]
            off_x, off_y = offsets[action]
            new_pos = self.walker_pos[0] + off_x, self.walker_pos[1] + off_y
            if self.grid[new_pos] == GridAbstraction.GRID_WALL:
                return self
            locked_door_positions = [(col, row) for _, col, row, _, state in self._doors if state > 0]
            if new_pos in locked_door_positions:
                return self
            return GridAbstraction(self.grid, new_pos, self.objects)
        if action == GridAbstraction.ACTION_GRAB:
            held_object = self._held_object
            if held_object == -1:
                object_positions = [(col, row) for id, col, row, _, _ in self.objects if id != GridAbstraction.DOOR_IDX]
                try:
                    held_object = object_positions.index(self.walker_pos)
                except ValueError:
                    # no object to pick up
                    return self
            new_objects = self.objects.copy()
            idx, col, row, color, _ = new_objects[held_object]
            new_objects[held_object] = (idx, col, row, color, int(self._held_object == -1))
            return GridAbstraction(self.grid, self.walker_pos, new_objects)
        
    def get_neighbors(self):
        neighbors = set([self.do_action(action) for action in GridAbstraction.ACTIONS])
        if self in neighbors:
            neighbors.remove(self)
        return neighbors

    @property
    def walker_grid_cell(self):
        # print(self.walker_pos)
        return self.grid[self.walker_pos]
    
    @property
    def grid_width(self):
        return self.grid.shape[0]
    
    @property
    def grid_height(self):
        return self.grid.shape[1]

    @staticmethod
    def grid_distance_from_state(grid_pos, minimujo_state: MinimujoState):
        wx = minimujo_state.pose[0] / minimujo_state.xy_scale
        wy = -minimujo_state.pose[1] / minimujo_state.xy_scale

        def clamp(x, a, b):
            return max(a, min(x, b))
        tx = clamp(wx, grid_pos[0], grid_pos[0]+1)
        ty = clamp(wy, grid_pos[1], grid_pos[1]+1)

        return (wx - tx)**2 + (wy - ty)**2
    
    @staticmethod
    def from_minimujo_state(minimujo_state: MinimujoState):
        def obj_to_grid(object_state: np.ndarray):
            id = object_state[0]
            col, row = GridAbstraction.continuous_position_to_grid(object_state[1:4], minimujo_state.xy_scale)
            color = object_state[14]
            state = object_state[15]
            return int(id), col, row, int(color), int(state)

        objects = [obj_to_grid(obj) for obj in minimujo_state.objects]
        walker_pos = GridAbstraction.continuous_position_to_grid(minimujo_state.pose[:3], minimujo_state.xy_scale)
        
        return GridAbstraction(minimujo_state.grid, walker_pos, objects)

    @staticmethod
    def continuous_position_to_grid(pos, xy_scale):
        col = int(np.floor(pos[0] / xy_scale))
        row = int(np.floor(-pos[1] / xy_scale))
        return col, row
    
    @staticmethod
    def pretty_object(object_tuple: Tuple[int]):
        oid, col, row, color_id, state = object_tuple
        name = 'unknown'
        if 0 <= oid < len(GridAbstraction.OBJECT_IDS):
            name = GridAbstraction.OBJECT_IDS[oid]
        color = list(COLOR_MAP.keys())[color_id]
        return f'[{color} {name} at ({col},{row}): {state}]'
    
    @staticmethod
    def backward_neighbor_edges(state: GridAbstraction):
        # we assume that actions are bidirectional. This should be the case, except maybe when objects are in the same cell
        neighbors = state.get_neighbors()
        return [(neighbor, 1) for neighbor in neighbors]
    
def get_minimujo_goal_wrapper(env: gym.Env, env_id: str, cls=GoalWrapper):
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_minimujo_state(state)
    planner = DjikstraBackwardsPlanner(GridAbstraction.backward_neighbor_edges)
    if 'RandomObject' in env_id:
        def goal_fn(obs, abstract, _env):
            task = _env.unwrapped._task
            pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
            matches = re.search(pattern, task.description)

            if not matches:
                raise Exception(f"Task '{task}' does not meet specification for RandomObject")
            color = matches.group(1)
            color_idx = get_color_idx(color)
            class_name = matches.group(2)
            class_idx = ['Ball','Box'].index(class_name)
            x = int(matches.group(3))
            y = int(matches.group(4))
            objects = abstract.objects.copy()
            for idx, obj in enumerate(abstract.objects):
                if obj[0] == class_idx and obj[3] == color_idx:
                    objects[idx] = (obj[0], x, y, obj[3], 0)

            return GridAbstraction(abstract.grid, (x, y), objects)
        def goal_obs_fn(abstract):
            obj_pos = abstract.objects[abstract._held_object][1:3] if abstract._held_object >= 0 else (-1, -1)
            return (*abstract.walker_pos, *obj_pos, abstract._held_object)
        low = [0, 0, 0, 0, 0]
        high = [5, 5, 5, 5, 5]
    else:
        def goal_fn(obs, abstract, _env):
            goal_idx = np.where(abstract.grid == GridAbstraction.GRID_GOAL)
            goal_x, goal_y = goal_idx[0][0], goal_idx[1][0]
            return GridAbstraction(abstract.grid, (goal_x, goal_y), abstract.objects)
        goal_obs_fn = lambda abstract: abstract.walker_pos
        low = [0, 0]
        high = [np.inf, np.inf]
    
    observer = GoalObserver(goal_obs_fn, low, high)
    return cls(env, abstraction_fn, goal_fn, planner, observer)

def get_minimujo_goal_wrapper_2(env: gym.Env, env_id: str, cls=GoalWrapper):
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_minimujo_state(state)
    planner = DeterministicValueIterationPlanner(GridAbstraction.ACTIONS, lambda state, action: state.do_action(action))
    if 'RandomObject' in env_id:
        def task_fn(obs, abstract, _env):
            """This returns a function that checks if the specified object is delivered to the goal"""
            task = _env.unwrapped._task
            pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
            matches = re.search(pattern, task.description)

            if not matches:
                raise Exception(f"Task '{task}' does not meet specification for RandomObject")
            color = matches.group(1)
            color_idx = get_color_idx(color)
            class_name = matches.group(2)
            class_idx = ['Ball','Box'].index(class_name)
            x = int(matches.group(3))
            y = int(matches.group(4))
            target = (class_idx, x, y, color_idx, 0)
            def task_status(abstract):
                if abstract.walker_grid_cell == GridAbstraction.GRID_LAVA:
                    return -1, True
                for obj in abstract.objects:
                    if obj == target:
                        return 1, True
                return 0, False
            return task_status
        
        def goal_obs_fn(abstract):
            obj_pos = abstract.objects[abstract._held_object][1:3] if abstract._held_object >= 0 else (-1, -1)
            return (*abstract.walker_pos, *obj_pos, abstract._held_object)
        low = [0, 0, 0, 0, 0]
        high = [5, 5, 5, 5, 5]
    else:
        def task_fn(obs, abstract, _env):
            def test_status(abstract):
                if abstract.walker_grid_cell == GridAbstraction.GRID_LAVA:
                    return -1, True
                if abstract.walker_grid_cell == GridAbstraction.GRID_GOAL:
                    return 1, True
                return 0, False
            return test_status
        
        goal_obs_fn = lambda abstract: abstract.walker_pos
        low = [0, 0]
        high = [np.inf, np.inf]
    
    observer = GoalObserver(goal_obs_fn, low, high)
    return cls(env, abstraction_fn, task_fn, planner, observer)
    
class GridSolver:

    def __init__(self):
        self.grid_state = None

    def solve_state(self, grid_state: GridAbstraction):
        self.grid_state = grid_state
        self.component_connections = self.get_all_connections()
        # print(self.component_connections)
        # print(list(map(lambda x: GridAbstraction.pretty_object(x), self.component_connections.keys())))
        

    def get_all_connections(self):
        self._agent = (-1, *self.grid_state.walker_pos, 0, 0)
        self._components = [self._agent, *self.grid_state.objects]
        # objects = []
        # for i in range(len(self.minigrid.grid.grid)):
        #     r, c = self.get_row_col(i)
        #     for world_obj in self.get_world_objects(i):
        #         if type(world_obj) in MinigridSolver.object_types:
        #             # c, r = world_obj.cur_pos
        #             world_obj.cur_pos = c, r
        #             objects.append((self.get_flat_idx(r, c), world_obj))
        all_connections = { component:self.connected_components(component[1:3]) for component in self._components}
        return all_connections

    def connected_components(self, start_pos):
        """Find connections to all components from starting position"""

        to_visit = deque([(start_pos, 0)])
        expanded = set([start_pos])
        grid_to_from = dict()
        grid_to_from[start_pos] = start_pos
        connected = []

        while len(to_visit) > 0:
            search_pos, dist = to_visit.popleft()
            grid_dir = self.get_grid_dir(grid_to_from[search_pos], search_pos)
            neighbors = self.get_neighbors(search_pos, dir=grid_dir)
            for neighbor_pos in neighbors:
                if neighbor_pos in expanded:
                    continue
                expanded.add(neighbor_pos)
                grid_to_from[neighbor_pos] = search_pos
                
                is_blocking = False
                for component in self._components:
                    if component[1:3] == neighbor_pos:
                        path_back = self.get_path_back(grid_to_from, start_pos, neighbor_pos)
                        connected.append((component, dist + 1, path_back))
                        if component[0] == GridAbstraction.DOOR_IDX and component[4] == GridAbstraction.DOOR_STATE_CLOSED:
                            is_blocking = True

                cell_value = self.grid_state.grid[neighbor_pos]
                if cell_value == GridAbstraction.GRID_WALL or cell_value == GridAbstraction.GRID_LAVA:
                    # not blocking
                    is_blocking = True

                if not is_blocking:
                    to_visit.append((neighbor_pos, dist + 1))
        return connected
    
    def get_grid_dir(self, from_pos, to_pos):
        diff = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        if diff[0] >= 1:
            # then to_idx is to the right
            return 0
        if diff[1] >= 1:
            # to_idx is below
            return 1
        if diff[0] <= -1:
            # to_idx is to the left
            return 2
        # to_idx is above
        return 3

    def get_neighbors(self, pos, dir=0):
        width = self.grid_state.grid_width
        col, row = pos
        neighbors = [None, None, None, None]
        if col + 1 < width:
            # right
            neighbors[0] = (col+1, row)
        if row + 1 < self.grid_state.grid_height:
            # down
            neighbors[1] = (col, row+1)
        if col - 1 >= 0:
            # left
            neighbors[2] = (col-1, row)
        if row - 1 >= 0:
            # up
            neighbors[3] = (col, row-1)
        # prioritize dir, then adjacent, and lastly the opposite dir
        prioritized = [neighbors[dir], neighbors[dir-3], neighbors[dir-1], neighbors[dir-2]]
        return [p for p in prioritized if p is not None]
    
    def get_path_back(self, grid_to_from, start_pos, end_pos):
        pos = end_pos
        path = [end_pos]
        while pos != start_pos:
            pos = grid_to_from[pos]
            path.append(pos)
            if len(path) > 1000:
                print("Path back has cycle")
                exit()
        return path

md5_kwargs = {}
if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    md5_kwargs = {'usedforsecurity': False}

# hashing utils
def hash_ndarray(arr):
    """Compute an MD5 hash for a NumPy array."""
    arr_bytes = arr.tobytes()
    hash_obj = hashlib.md5(arr_bytes, **md5_kwargs)
    return hash_obj.hexdigest()
    
def hash_list_of_tuples(list_of_tuples):
    """Compute a hash for a list of tuples."""
    list_bytes = pickle.dumps(list_of_tuples)
    hash_obj = hashlib.md5(list_bytes, **md5_kwargs)
    return hash_obj.hexdigest()

# if __name__ == "__main__":
#     import gymnasium as gym
#     from minimujo.state.minimujo_state import MinimujoStateObserver
#     minigrid_id = "MiniGrid-Empty-5x5-v0"
#     # minigrid_id = "MiniGrid-Playground-v0"
#     minigrid_env = gym.make(minigrid_id, render_mode='human').unwrapped
#     minigrid_env.reset()
#     grid = MinimujoStateObserver.get_grid_state_from_minigrid(minigrid_env)
#     row, col = minigrid_env.agent_pos
#     GridAbstraction(grid, (col, row))