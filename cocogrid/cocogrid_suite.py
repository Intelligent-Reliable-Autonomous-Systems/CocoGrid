"""Define a suite of Cocogrid environment constructors."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium
from dm_control import composer
from dm_control.locomotion.walkers import ant, cmu_humanoid
from dm_control.locomotion.walkers.base import Walker
from dm_control.utils import containers
from gymnasium.envs.registration import registry
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from gymnasium.wrappers.order_enforcing import OrderEnforcing

from cocogrid.custom_minigrid import register_custom_minigrid
from cocogrid.dmc_gym import CocogridGym
from cocogrid.mujoco import CocogridArena, CocogridTask
from cocogrid.multitask import register_multitask_minigrid
from cocogrid.state.tasks import DEFAULT_TASK_REGISTRY, get_null_task
from cocogrid.walkers.rolling_ball import RollingBallWithHead
from cocogrid.walkers.square import Square

if TYPE_CHECKING:
    from cocogrid.state import ObservationSpecification


def get_cocogrid_env(
    minigrid_id: str,
    walker_type: str = "square",
    timesteps: int = 500,
    seed: int | None = None,
    environment_kwargs: dict | None = None,
) -> composer.Environment:
    """Construct a dm_control environment."""
    if "minigrid" in environment_kwargs:
        high_env: gymnasium.Env = environment_kwargs.pop("minigrid")
    else:
        high_env = gymnasium.make(minigrid_id, disable_env_checker=True).env
        if isinstance(high_env, PassiveEnvChecker):
            high_env = high_env.env
        if isinstance(high_env, OrderEnforcing):
            high_env = high_env.env
    high_env.reset(seed=seed)

    environment_kwargs = environment_kwargs or {}
    task_kwargs = {}
    task_keys = [
        "observation_type",
        "random_rotation",
        "task_function",
        "get_task_function",
    ]
    for key in task_keys:
        if key in environment_kwargs:
            task_kwargs[key] = environment_kwargs.pop(key)

    if "reward_type" in environment_kwargs:
        print("Deprecation Warning: reward_type is unused. specify the reward in the task function")
        environment_kwargs.pop("reward_type")

    if walker_type == "rolling_ball" or walker_type == "ball":
        walker = RollingBallWithHead(initializer=())
    elif walker_type == "cmu_humanoid" or walker_type == "human" or walker_type == "humanoid":
        walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
            observable_options={"egocentric_camera": {"enabled": True}}
        )
    elif walker_type == "ant":
        walker = ant.Ant()
        # walker = Ant()
        environment_kwargs["xy_scale"] = max(2, environment_kwargs.get("xy_scale", 2))
        environment_kwargs["spawn_padding"] = 0.8
    elif walker_type == "square":
        walker = Square()
        task_kwargs["random_rotation"] = False
    elif isinstance(walker_type, Walker):
        walker = walker_type
    else:
        raise Exception(f"walker_type {walker_type} not supported")

    environment_kwargs["seed"] = seed

    arena = CocogridArena(high_env, **environment_kwargs)

    physics_timestep = 0.005
    control_timestep = 0.03
    time_limit = control_timestep * timesteps - 0.00001  # subtract a tiny amount due to precision error

    if "task_function" in task_kwargs:
        # the task function is the same every episode
        task_function = task_kwargs.pop("task_function")
        task_kwargs["get_task_function"] = lambda minigrid: (task_function, "")
    if "get_task_function" not in task_kwargs:
        task_kwargs["get_task_function"] = DEFAULT_TASK_REGISTRY.get(type(high_env), get_null_task)
    task = CocogridTask(
        walker=walker,
        cocogrid_arena=arena,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        contact_termination=False,
        **task_kwargs,
    )

    return composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
    )


def get_gym_env_from_suite(
    domain: str,
    task: str,
    walker_type: str = "ball",
    observation_type: str | ObservationSpecification = "no-arena",
    timesteps: int = 200,
    seed: int | None = None,
    render_mode: str = "rgb_array",
    render_width: int = 64,
    **env_kwargs: dict,
) -> gymnasium.Env:
    """Construct a gymnasium environment specified by the suite."""
    if "box2d" in walker_type.lower():
        return get_box2d_gym_env(
            task,
            walker_type,
            observation_type=observation_type,
            timesteps=timesteps,
            seed=seed,
            render_width=render_width,
            **env_kwargs,
        )
    return CocogridGym(
        domain=domain,
        task=task,
        task_kwargs={"walker_type": walker_type, "timesteps": timesteps, "seed": seed},
        environment_kwargs=env_kwargs,
        observation_type=observation_type,
        rendering=None,
        render_width=render_width,
        render_mode=render_mode,
    )


def get_box2d_gym_env(
    arena_id: str,
    walker_type: str,
    observation_type: str | ObservationSpecification = None,
    task_function: callable | None = None,
    get_task_function: callable | None = None,
    minigrid: gymnasium.Env | None = None,
    **env_kwargs: dict,
) -> gymnasium.Env:
    """Construct a Box2D gymnasium environment."""
    from cocogrid.box2d.gym import Box2DEnv

    if minigrid is not None:
        minigrid_env = minigrid
    else:
        minigrid_id = arena_id.replace("Cocogrid", "MiniGrid")
        minigrid_env = gymnasium.make(minigrid_id, disable_env_checker=True).env
        if isinstance(minigrid_env, OrderEnforcing):
            minigrid_env = minigrid_env.env

    if task_function is not None:
        # the task function is the same every episode
        def get_task_function(minigrid: gymnasium.Env) -> tuple[callable, str]:
            return (task_function, "")
    elif get_task_function is None:
        get_task_function = DEFAULT_TASK_REGISTRY.get(type(minigrid_env), get_null_task)
    return Box2DEnv(
        minigrid_env,
        walker_type,
        get_task_function,
        observation_type=observation_type,
        **env_kwargs,
    )


SUITE = containers.TaggedTasks()

register_custom_minigrid()
register_multitask_minigrid()
minigrid_env_ids = [env_spec.id for env_spec in registry.values() if env_spec.id.startswith("MiniGrid")]

for minigrid_id in minigrid_env_ids:
    cocogrid_id = minigrid_id.replace("MiniGrid", "Cocogrid")
    SUITE._tasks[cocogrid_id] = functools.partial(get_cocogrid_env, minigrid_id)


def get_cocogrid_env_ids() -> list[str]:
    """Get a list of environment ids registered in the suite."""
    return list(SUITE._tasks.keys())
