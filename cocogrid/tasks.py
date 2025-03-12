from __future__ import annotations

import inspect
import re
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.minigrid_env import MiniGridEnv

from cocogrid.common.abstraction.grid_abstraction import GridAbstraction
from cocogrid.common.cocogrid_state import CocogridState
from cocogrid.common.entity import OBJECT_NAMES, ObjectEnum, get_color_id
from cocogrid.common.registry import NotRegisteredError

TaskEvalType = Callable[[CocogridState, CocogridState], tuple[float, bool]]
TaskGetterType = Callable[[MiniGridEnv], tuple[TaskEvalType, str]]


class TaskRegistry:
    """A registry for tasks."""

    _instance: TaskRegistry | None = None
    _initialized: bool = False

    def __new__(cls) -> TaskRegistry:
        """Construct a TaskRegistry instance if necessary."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> TaskRegistry:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Construct a task registry."""
        if TaskRegistry._initialized:
            return
        self._env_task_registry: dict[type[gym.Env], TaskGetterType | TaskEvalType] = {type[gym.Env]: get_null_task}
        TaskRegistry._initialized = True

    def register_env_task(self, cls: type[gym.Env], value: TaskGetterType | TaskEvalType) -> None:
        """Register a value for a specific class."""
        self._env_task_registry[cls] = value

    def get_task_for_env(self, cls: type[gym.Env]) -> TaskGetterType:
        """Get the task getter associated with an environment type."""
        task = None
        for base_cls in cls.mro():  # Traverse class hierarchy (Method Resolution Order)
            if base_cls in self._env_task_registry:
                task = self._env_task_registry[base_cls]
                break
        if task is not None:
            task_getter = task
            if self._is_task_eval_type(task):
                def get_task(minigrid: MiniGridEnv) -> tuple[TaskEvalType, str]:
                    return task, task.__name__

                task_getter = get_task
            return task_getter

        raise NotRegisteredError("TaskRegistry", str(cls))

    def _is_task_eval_type(self, task: TaskEvalType | TaskGetterType) -> bool:
        if not callable(task):
            return False

        sig = inspect.signature(task)
        params = list(sig.parameters.values())

        return (
            len(params) == 2 and  # Must have exactly two parameters
            all(p.annotation in (CocogridState, inspect.Parameter.empty) for p in params) and  # Parameter types match
            sig.return_annotation in (tuple[float, bool], inspect.Signature.empty)  # Return type matches
        )


def get_null_task(minigrid_env: MiniGridEnv) -> tuple[TaskEvalType, str]:
    """Get an empty task with no success or failure conditions."""

    def no_task_function(prev_state: CocogridState, cur_state: CocogridState) -> tuple[float, bool]:
        """Give no reward and never terminate always."""
        return 0, False

    return no_task_function, "No task."


def get_grid_goal_task(minigrid: MiniGridEnv) -> tuple[TaskEvalType, str]:
    """Get the task for reaching a goal."""

    def grid_goal_task(prev_state: CocogridState, cur_state: CocogridState) -> tuple[float, bool]:
        """Evaluate when a goal has been reached, or fails due to lava."""
        grid_state = GridAbstraction.from_cocogrid_state(cur_state)
        cell_value = grid_state.walker_grid_cell
        if cell_value == GridAbstraction.GRID_GOAL:
            return 1, True
        if cell_value == GridAbstraction.GRID_LAVA:
            return -1, True
        return 0, False

    return grid_goal_task, minigrid.mission


def get_pickup_task(minigrid_env: MiniGridEnv, strict: bool = False) -> tuple[TaskEvalType, str]:
    """Get the task for picking up an object of a certain color specified by minigrid_env.mission.

    If strict, then penalize picking up the wrong object.
    """
    target_color = extract_feature_from_mission_text(minigrid_env.mission, COLOR_NAMES)
    color_id = get_color_id(target_color)
    target_obj_type = extract_feature_from_mission_text(minigrid_env.mission, OBJECT_NAMES)
    obj_id = ObjectEnum.get_id(target_obj_type)

    def pickup_task(prev_state: CocogridState, cur_state: CocogridState) -> tuple[float, bool]:
        """Evaluate when an object has been picked up."""
        for obj in cur_state.objects:
            obj_type = obj[CocogridState.OBJECT_IDX_TYPE]
            # if is a held object
            if obj_type != ObjectEnum.DOOR.value and obj[CocogridState.OBJECT_IDX_STATE] == 1:
                # if matches target
                if obj[CocogridState.OBJECT_IDX_COLOR] == color_id and obj_type == obj_id:
                    return 1, True
                if strict:
                    # in strict mode, only allow picking up target
                    return -1, True
        return 0, False

    return pickup_task, minigrid_env.mission


def get_put_near_task(minigrid_env: MiniGridEnv) -> tuple[TaskEvalType, str]:
    """Get the task for putting two objects specified in minigrid_env.mission within one unit of each other."""
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

    def put_near_task(prev_state: CocogridState, cur_state: CocogridState) -> tuple[float, bool]:
        """Evaluate when the objects are near each other."""
        for idx1, obj1 in enumerate(cur_state.objects):
            if obj1[CocogridState.OBJECT_IDX_COLOR] != cidx1 or obj1[CocogridState.OBJECT_IDX_TYPE] != tidx1:
                continue
            pos1 = obj1[pos_idx : pos_idx + 3]
            vel1 = obj1[vel_idx : vel_idx + 3]
            if np.linalg.norm(vel1) > 0.1:
                continue
            for idx2, obj2 in enumerate(cur_state.objects):
                if idx1 == idx2:
                    continue
                if obj2[CocogridState.OBJECT_IDX_COLOR] != cidx2 or obj2[CocogridState.OBJECT_IDX_TYPE] != tidx2:
                    continue
                pos2 = obj2[pos_idx : pos_idx + 3]
                vel2 = obj2[vel_idx : vel_idx + 3]
                if np.linalg.norm(vel2) > 0.1:
                    continue
                if np.linalg.norm(pos1 - pos2) < 1:
                    return 1, True
        return 0, False

    return put_near_task, minigrid_env.mission


def get_strict_pickup_task(minigrid_env: MiniGridEnv) -> tuple[TaskEvalType, str]:
    """Wrap get_pickup_task with strict=True."""
    return get_pickup_task(minigrid_env, strict=True)


def infer_task(minigrid_env: MiniGridEnv) -> tuple[TaskEvalType, str]:
    """Dynamically infer the task from the environment class."""
    minigrid_env = minigrid_env.unwrapped
    task_func = TaskRegistry.get_instance().get_task_for_env(type(minigrid_env))
    return task_func(minigrid_env)


def extract_feature_from_mission_text(mission: str, features: list[str], index: int = 0) -> str:
    """Extract patterns from the mission text.

    Input:
    mission -- a text string describing the mission.
    features -- a list of keywords to try extracting.
    index -- if a mission text contains multiple features, index which one to extract.

    Output:
    One of the keywords from the features extracted from mission.

    Exceptions:
    ValueError if mission text does not contain at least #index features.
    """
    # The features can be colors ["red", "blue", "yellow", etc.] or object types ["ball", "box", "key", "door"]
    pattern = r"\b(?:" + "|".join(re.escape(feature) for feature in features) + r")\b"
    matches = re.findall(pattern, mission, re.IGNORECASE)
    if len(matches) < index + 1:
        raise ValueError(f"For mission '{mission}', could not extract from {features} (index {index})")
    return matches[index].lower()
