from __future__ import annotations
from typing import Any, Callable

from gymnasium.envs.registration import register
from minigrid import envs

from cocogrid.common.multitask import MultiTaskBuilder, MultiTaskEnv
from cocogrid.minigrid.doorkeycrossing import DoorKeyCrossingEnv
from cocogrid.minigrid.hallwaychoice import HallwayChoiceEnv
from cocogrid.minigrid.objectdelivery import ObjectDeliveryEnv
from cocogrid.minigrid.randomcorner import RandomCornerEnv
from cocogrid.minigrid.umaze import UMazeEnv as UMazeEnv
from cocogrid.tasks import (
    TaskRegistry,
    get_grid_goal_task,
    get_null_task,
    get_pickup_task,
    get_put_near_task,
    get_strict_pickup_task,
    infer_task,
)


def register_base_minigrid_tasks() -> None:
    """Register tasks for the environments provided in the base minigrid package."""
    registry = TaskRegistry.get_instance()

    registry.register_env_task(envs.FetchEnv, get_strict_pickup_task)
    registry.register_env_task(envs.KeyCorridorEnv, get_pickup_task)
    registry.register_env_task(envs.UnlockPickupEnv, get_pickup_task)
    registry.register_env_task(envs.BlockedUnlockPickupEnv, get_pickup_task)
    registry.register_env_task(envs.CrossingEnv, get_grid_goal_task)
    registry.register_env_task(envs.DoorKeyEnv, get_grid_goal_task)
    registry.register_env_task(envs.FourRoomsEnv, get_grid_goal_task)
    registry.register_env_task(envs.EmptyEnv, get_grid_goal_task)
    registry.register_env_task(envs.DistShiftEnv, get_grid_goal_task)
    registry.register_env_task(envs.LavaGapEnv, get_grid_goal_task)
    registry.register_env_task(envs.LockedRoomEnv, get_grid_goal_task)
    registry.register_env_task(envs.MultiRoomEnv, get_grid_goal_task)
    registry.register_env_task(envs.PlaygroundEnv, get_null_task)
    registry.register_env_task(envs.PutNearEnv, get_put_near_task)

    registry.register_env_task(MultiTaskEnv, infer_task)


CUSTOM_MINIGRID_ENVS = []


def register_custom_minigrid_envs() -> list[str]:
    """Register custom Minigrid environments and return the new gym ids."""
    if len(CUSTOM_MINIGRID_ENVS) > 0:
        # already registered
        return CUSTOM_MINIGRID_ENVS

    def register_and_log(id: str, entry_point: str | Callable, kwargs: dict[str, Any] | None = None) -> None:
        if kwargs is None:
            kwargs = {}
        register(id=id, entry_point=entry_point, kwargs=kwargs)
        CUSTOM_MINIGRID_ENVS.append(id)

    register_and_log(id="MiniGrid-UMaze-v0", entry_point="cocogrid.minigrid.umaze:UMazeEnv")
    register_and_log(id="MiniGrid-RandomCorner-v0", entry_point="cocogrid.minigrid.randomcorner:RandomCornerEnv")
    register_and_log(id="MiniGrid-HallwayChoice-v0", entry_point="cocogrid.minigrid.hallwaychoice:HallwayChoiceEnv")
    register_and_log(
        id="MiniGrid-ObjectDelivery-v0",
        entry_point="cocogrid.minigrid.objectdelivery:ObjectDeliveryEnv",
        kwargs={"num_objects": 1},
    )
    register_and_log(
        id="MiniGrid-ObjectDelivery-3-v0",
        entry_point="cocogrid.minigrid.objectdelivery:ObjectDeliveryEnv",
        kwargs={"num_objects": 3},
    )
    register_and_log(
        id="MiniGrid-DoorKeyCrossingS9N3-v0",
        entry_point="cocogrid.minigrid.doorkeycrossing:DoorKeyCrossingEnv",
        kwargs={"size": 9, "num_crossings": 3},
    )
    # register_and_log(
    #     id='MiniGrid-RandomObjects-3-yellow-green-v0',
    #     entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
    #     kwargs={
    #         'num_objects': 3,
    #         'colors': ['yellow', 'green']
    #     }
    # )
    # register(
    #     id='MiniGrid-RandomObjects-3-goal-left-v0',
    #     entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
    #     kwargs={
    #         'num_objects': 3,
    #         'goal_positions': [(1,1), (1,2),(1,3),(2,1)],
    #         'obj_positions': [(3,1), (3,2), (3,3), (2,3)]
    #     }
    # )
    # register(
    #     id='MiniGrid-RandomObjects-3-goal-right-color-v0',
    #     entry_point='cocogrid.custom_minigrid:RandomObjectsEnv',
    #     kwargs={
    #         'num_objects': 3,
    #         'goal_positions': [(3,1), (3,2), (3,3), (2,3)],
    #         'obj_positions': [(1,1), (1,2),(1,3),(2,1)],
    #         'colors': ['yellow', 'green', 'purple', 'grey']
    #     }
    # )

    # Register some extra parameterizations of existing environments.
    register_and_log(
        id="MiniGrid-DoorKey-10x10-v0", entry_point="minigrid.envs.doorkey:DoorKeyEnv", kwargs={"size": 10}
    )
    register_and_log(
        id="MiniGrid-DoorKey-12x12-v0", entry_point="minigrid.envs.doorkey:DoorKeyEnv", kwargs={"size": 12}
    )

    def multi_goal(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-Empty-5x5-v0", size=size) \
            .add_env("MiniGrid-RandomCorner-v0", size=size) \
            .add_env("MiniGrid-HallwayChoice-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N1-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N3-v0", size=size)
        return multitask.build_env()

    def multi_goal_eval_easier(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_eval("MiniGrid-Empty-5x5-v0", size=size) \
            .add_eval("MiniGrid-RandomCorner-v0", size=size) \
            .add_eval("MiniGrid-HallwayChoice-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N1-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N3-v0", size=size)
        return multitask.build_env()

    def multi_goal_eval_harder(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-Empty-5x5-v0", size=size) \
            .add_env("MiniGrid-RandomCorner-v0", size=size) \
            .add_env("MiniGrid-HallwayChoice-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N1-v0", size=size) \
            .add_eval("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_eval("MiniGrid-SimpleCrossingS9N3-v0", size=size)
        return multitask.build_env()

    def multi_goal_eval_hard1(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-Empty-5x5-v0", size=size) \
            .add_env("MiniGrid-RandomCorner-v0", size=size) \
            .add_env("MiniGrid-HallwayChoice-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N1-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N3-v0", size=size) \
            .add_eval("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_eval("MiniGrid-SimpleCrossingS9N3-v0", size=size)
        return multitask.build_env()

    def multi_goal_eval_hard2(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-SimpleCrossingS9N2-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N3-v0", size=size)
        return multitask.build_env()

    register_and_log(
        id="MiniGrid-MultiGoal-7x7-v0",
        entry_point=lambda: multi_goal(size=7),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-9x9-v0",
        entry_point=lambda: multi_goal(size=9),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-EvalEasier-7x7-v0",
        entry_point=lambda: multi_goal_eval_easier(size=7),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-EvalEasier-9x9-v0",
        entry_point=lambda: multi_goal_eval_easier(size=9),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-EvalHarder-7x7-v0",
        entry_point=lambda: multi_goal_eval_harder(size=7),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-TrainAll-EvalHard-7x7-v0",
        entry_point=lambda: multi_goal_eval_hard1(size=7),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-TrainHard-EvalHard-7x7-v0",
        entry_point=lambda: multi_goal_eval_hard2(size=7),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-EvalHarder-9x9-v0",
        entry_point=lambda: multi_goal_eval_harder(size=9),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-TrainAll-EvalHard-9x9-v0",
        entry_point=lambda: multi_goal_eval_hard1(size=9),
    )

    register_and_log(
        id="MiniGrid-MultiGoal-TrainHard-EvalHard-9x9-v0",
        entry_point=lambda: multi_goal_eval_hard2(size=9),
    )

    def doorkey_crossing_generalization(size=9, num_crossings=1, obstacle_type=None, **kwargs):
        crossing_args = {
            "size": size,
            "num_crossings": num_crossings,
        }
        if obstacle_type is not None:
            crossing_args["obstacle_type"] = obstacle_type
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-DoorKey-6x6-v0", size=size) \
            .add_env("MiniGrid-SimpleCrossingS9N2-v0", **crossing_args) \
            .add_eval("MiniGrid-DoorKeyCrossingS9N3-v0", **crossing_args)
        return multitask.build_env()

    register_and_log(
        id="MiniGrid-DoorKeyCrossingGeneralizationS9N3-v0",
        entry_point=lambda: doorkey_crossing_generalization(size=9, num_crossings=3),
    )

    def randobjs3_color_generalize():
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-RandomObjects-3-v0") \
            .add_eval("MiniGrid-RandomObjects-3-yellow-green-v0")
        return multitask.build_env()
    register_and_log(
        id="MiniGrid-RandomObjects-3-color-generalize-v0",
        entry_point=randobjs3_color_generalize,
    )

    def randobjs3_color_position_generalize():
        multitask = MultiTaskBuilder() \
            .add_env("MiniGrid-RandomObjects-3-goal-left-v0") \
            .add_eval("MiniGrid-RandomObjects-3-goal-right-color-v0")
        return multitask.build_env()
    register_and_log(
        id="MiniGrid-RandomObjects-3-color-pos-generalize-v0",
        entry_point=randobjs3_color_position_generalize,
    )

    return CUSTOM_MINIGRID_ENVS


def register_custom_minigrid_tasks() -> None:
    """Register cocogrid tasks for custom minigrid environments."""
    registry = TaskRegistry.get_instance()

    registry.register_env_task(DoorKeyCrossingEnv, get_grid_goal_task)
    registry.register_env_task(HallwayChoiceEnv, get_grid_goal_task)
    registry.register_env_task(ObjectDeliveryEnv, ObjectDeliveryEnv.get_cocogrid_task)
    registry.register_env_task(RandomCornerEnv, get_grid_goal_task)
    registry.register_env_task(UMazeEnv, get_grid_goal_task)

__all__ = [
    register_base_minigrid_tasks,
    register_custom_minigrid_envs,
    register_custom_minigrid_tasks,
    CUSTOM_MINIGRID_ENVS,
]
