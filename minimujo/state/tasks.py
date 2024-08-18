

import numpy as np
from minimujo.color import get_color_idx
from minimujo.state.grid_abstraction import GridAbstraction
from minimujo.state.minimujo_state import MinimujoState
from minigrid.core.world_object import Box
from minigrid.minigrid_env import MiniGridEnv


def get_grid_goal_task(minigrid: MiniGridEnv):
    def grid_goal_task(prev_state: MinimujoState, cur_state: MinimujoState):
        grid_state = GridAbstraction.from_minimujo_state(cur_state)
        cell_value = grid_state.walker_grid_cell
        if cell_value == GridAbstraction.GRID_GOAL:
            return 1, True
        if cell_value == GridAbstraction.GRID_LAVA:
            return -1, True
        return 0, False
    return grid_goal_task, "Agent should reach goal tile"

def get_random_objects_task(random_objects_env):
    """Gets the task for the RandomObjectsEnv"""
    color, cls, x, y = random_objects_env.target
    color_idx = get_color_idx(color)
    cls_idx = 1 if cls == Box else 0
    target_object = (cls_idx, x, y, color_idx, 0)
    
    def object_to_position_task(prev_state: MinimujoState, cur_state: MinimujoState):
        grid_state = GridAbstraction.from_minimujo_state(cur_state)
        for idx, obj in enumerate(grid_state.objects):
            if obj == target_object:
                cvel = cur_state.objects[idx,8:11]
                vel = np.linalg.norm(cvel)
                if vel < 0.1:
                    return 1, True
        return 0, False
    return object_to_position_task, f"Deliver a {color} {cls.__name__} to tile ({x}, {y})."