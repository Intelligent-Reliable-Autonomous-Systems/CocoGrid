from __future__ import annotations
import hashlib
import pickle
import sys
from typing import List, Tuple
import numpy as np
from minimujo.state.minimujo_state import MinimujoState


class GridAbstraction:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    ACTION_GRAB = 4
    DOOR_IDX = 2
    WALL_IDX = 1
    GOAL_IDX = 2
    LAVA_IDX = 3

    def __init__(self, grid: np.ndarray, walker_pos: Tuple[int], objects: List[Tuple[int]]) -> None:
        self.grid: np.ndarray = grid
        self._grid_hash: str = hash_ndarray(self.grid)

        self.walker_pos = walker_pos
        
        self.objects = objects.copy()
        self._doors = [obj for obj in self.objects if obj[0] == GridAbstraction.DOOR_IDX]
        try:
            self._holdable_objects = [state for idx, _, _, _, state in self.objects if idx != GridAbstraction.DOOR_IDX]
            self._held_object = self._holdable_objects.index(1)
        except ValueError:
            self._held_object = -1
        if self._held_object > -1:
            # held object should be snapped to walker position
            idx, _, _, color, state = self.objects[self._held_object]
            self.objects[self._held_object] = (idx, walker_pos[0], walker_pos[1], color, state)
        self._objects_hash = hash_list_of_tuples(self.objects)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridAbstraction):
            return False
        return self.walker_pos == other.walker_pos and \
            self._grid_hash == other._grid_hash and \
            self._objects_hash == other._objects_hash
    
    def do_action(self, action: int) -> GridAbstraction:
        if action < 4:
            offsets = [(0,-1),(-1,0),(0,1),(1,0)]
            off_x, off_y = offsets[action]
            new_pos = self.walker_pos[0] + off_x, self.walker_pos[1] + off_y
            if self.grid[new_pos] == GridAbstraction.WALL_IDX:
                return self
            door_positions = [(col, row) for _, col, row, _, _ in self._doors]
            if new_pos in door_positions:
                return self
            return GridAbstraction(self.grid, new_pos, self.objects)
        if action == GridAbstraction.ACTION_GRAB:
            held_object = self._held_object
            if held_object == -1:
                object_positions = [(col, row) for _, col, row, _, _ in self._holdable_objects]
                try:
                    held_object = object_positions.index(self.walker_pos)
                except ValueError:
                    # no object to pick up
                    return self
            new_objects = self.objects.copy()
            idx, col, row, color, _ = new_objects[held_object]
            new_objects[held_object] = (idx, col, row, color, self._held_object == -1)

    @property
    def walker_grid_cell(self):
        # print(self.walker_pos)
        return self.grid[self.walker_pos]

    @staticmethod
    def grid_distance_from_state(grid_pos, minimujo_state: MinimujoState):
        wx = minimujo_state.pose[0] / minimujo_state.xy_scale
        wy = -minimujo_state.pose[1] / minimujo_state.xy_scale

        def clamp(x, a, b):
            return max(a, min(x, b))
        tx = clamp(wx, grid_pos[0], grid_pos[0]+1)
        ty = clamp(wx, grid_pos[1], grid_pos[1]+1)

        return (wx - tx)**2 + (wy - ty)**2
    
    @staticmethod
    def from_minimujo_state(minimujo_state: MinimujoState):
        def obj_to_grid(object_state: np.ndarray):
            id = object_state[0]
            col, row = GridAbstraction.continuous_position_to_grid(object_state[1:4], minimujo_state.xy_scale)
            color = object_state[14]
            state = object_state[15]
            return id, col, row, color, state

        objects = [obj_to_grid(obj) for obj in minimujo_state.objects]
        walker_pos = GridAbstraction.continuous_position_to_grid(minimujo_state.pose[:3], minimujo_state.xy_scale)
        
        return GridAbstraction(minimujo_state.grid, walker_pos, objects)

    @staticmethod
    def continuous_position_to_grid(pos, xy_scale):
        col = int(np.floor(pos[0] / xy_scale))
        row = int(np.floor(-pos[1] / xy_scale))
        return col, row

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