import re
from typing import Callable, Dict
import numpy as np
from minigrid.core.world_object import Goal
from cocogrid.entities import OBJECT_NAMES, ObjectEnum, get_color_id
from cocogrid.state.grid_abstraction import GridAbstraction
from cocogrid.state.cocogrid_state import CocogridState
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
import minigrid.envs as envs


def get_grid_goal_task(minigrid: MiniGridEnv):
    def grid_goal_task(prev_state: CocogridState, cur_state: CocogridState):
        grid_state = GridAbstraction.from_cocogrid_state(cur_state)
        cell_value = grid_state.walker_grid_cell
        if cell_value == GridAbstraction.GRID_GOAL:
            return 1, True
        if cell_value == GridAbstraction.GRID_LAVA:
            return -1, True
        return 0, False
    return grid_goal_task, minigrid.mission

def get_random_objects_task(random_objects_env):
    """Gets the task for the RandomObjectsEnv"""
    color, cls, x, y = random_objects_env.target
    color_idx = get_color_id(color)
    cls_idx = ObjectEnum.get_id(cls)
    target_object = (cls_idx, x, y, color_idx, 0)
    
    def object_to_position_task(prev_state: CocogridState, cur_state: CocogridState):
        grid_state = GridAbstraction.from_cocogrid_state(cur_state)
        for idx, obj in enumerate(grid_state.objects):
            if obj == target_object:
                cvel = cur_state.objects[idx,8:11]
                vel = np.linalg.norm(cvel)
                if vel < 0.1:
                    return 1, True
        return 0, False
    return object_to_position_task, f"Deliver a {color} {cls.__name__} to tile ({x}, {y})."

def get_pickup_task(minigrid_env: MiniGridEnv, strict: bool = False):
    target_color = extract_feature_from_mission_text(minigrid_env.mission, COLOR_NAMES)
    color_id = get_color_id(target_color)
    target_obj_type = extract_feature_from_mission_text(minigrid_env.mission, OBJECT_NAMES)
    obj_id = ObjectEnum.get_id(target_obj_type)

    def pickup_task(prev_state: CocogridState, cur_state: CocogridState):
        for obj in cur_state.objects:
            obj_type = obj[CocogridState.OBJECT_IDX_TYPE]
            # if is a held object
            if obj_type != ObjectEnum.DOOR.value and obj[CocogridState.OBJECT_IDX_STATE] == 1:
                # if matches target
                if obj[CocogridState.OBJECT_IDX_COLOR] == color_id and obj_type == obj_id:
                    return 1, True
                elif strict:
                    # in strict mode, only allow picking up target
                    return -1, True
        return 0, False
    return pickup_task, minigrid_env.mission

def get_put_near_task(minigrid_env: MiniGridEnv):
    color1 = extract_feature_from_mission_text(minigrid_env.mission, COLOR_NAMES, 0)
    target1 = extract_feature_from_mission_text(minigrid_env.mission, OBJECT_NAMES, 0)
    cidx1 = get_color_id(color1)
    tidx1 = ObjectEnum.get_id(target1)
    color2 = extract_feature_from_mission_text(minigrid_env.mission, COLOR_NAMES, 1)
    target2 = extract_feature_from_mission_text(minigrid_env.mission, OBJECT_NAMES, 1)
    cidx2 = get_color_id(color2)
    tidx2 = ObjectEnum.get_id(target2)
    pos_idx = CocogridState.OBJECT_IDX_POS
    vel_idx = CocogridState.OBJECT_IDX_VEL

    def put_near_task(prev_state: CocogridState, cur_state: CocogridState):
        for idx1, obj1 in enumerate(cur_state.objects):
            if obj1[CocogridState.OBJECT_IDX_COLOR] != cidx1 \
                or obj1[CocogridState.OBJECT_IDX_TYPE] != tidx1:
                continue
            pos1 = obj1[pos_idx:pos_idx+3]
            vel1 = obj1[vel_idx:vel_idx+3]
            if np.linalg.norm(vel1) > 0.1:
                continue
            for idx2, obj2 in enumerate(cur_state.objects):
                if idx1 == idx2:
                    continue
                if obj2[CocogridState.OBJECT_IDX_COLOR] != cidx2 \
                    or obj2[CocogridState.OBJECT_IDX_TYPE] != tidx2:
                    continue
                pos2 = obj2[pos_idx:pos_idx+3]
                vel2 = obj2[vel_idx:vel_idx+3]
                if np.linalg.norm(vel2) > 0.1:
                    continue
                if np.linalg.norm(pos1 - pos2) < 1:
                    return 1, True
        return 0, False
    return put_near_task, minigrid_env.mission

def get_strict_pickup_task(minigrid_env: MiniGridEnv):
    return get_pickup_task(minigrid_env, strict=True)

def get_null_task(minigrid_env: MiniGridEnv):
    def no_task_function(prev_state: CocogridState, cur_state: CocogridState):
        return 0, False
    return no_task_function, ""

def infer_task(minigrid_env: MiniGridEnv):
    minigrid_env = minigrid_env.unwrapped
    task_func = DEFAULT_TASK_REGISTRY.get(type(minigrid_env), get_null_task)
    return task_func(minigrid_env)

def extract_feature_from_mission_text(mission, features, index=0):
    # The features can be colors ["red", "blue", "yellow", etc.] or object types ["ball", "box", "key", "door"]
    pattern = r'\b(?:' + '|'.join(re.escape(feature) for feature in features) + r')\b'
    matches = re.findall(pattern, mission, re.IGNORECASE)
    if len(matches) < index + 1:
        raise ValueError(f"For mission '{mission}', could not extract from {features} (index {index})")
    return matches[index].lower()

DEFAULT_TASK_REGISTRY: Dict[MiniGridEnv, Callable] = {
    envs.FetchEnv: get_strict_pickup_task,
    envs.KeyCorridorEnv: get_pickup_task,
    envs.UnlockPickupEnv: get_pickup_task,
    envs.BlockedUnlockPickupEnv: get_pickup_task,
    envs.CrossingEnv: get_grid_goal_task,
    envs.DoorKeyEnv: get_grid_goal_task,
    envs.FourRoomsEnv: get_grid_goal_task,
    envs.EmptyEnv: get_grid_goal_task,
    envs.DistShiftEnv: get_grid_goal_task,
    envs.LavaGapEnv: get_grid_goal_task,
    envs.LockedRoomEnv: get_grid_goal_task,
    envs.MultiRoomEnv: get_grid_goal_task,
    envs.PlaygroundEnv: get_null_task,
    envs.PutNearEnv: get_put_near_task
}