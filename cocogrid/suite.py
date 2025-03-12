from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from cocogrid.agent import Agent, AgentRegistry

if TYPE_CHECKING:
    import gymnasium as gym
    from minigrid.minigrid_env import MiniGridEnv

    from cocogrid.common.observation import ObservationSpecification


def get_cocogrid_gymnasium_env(
    minigrid: str | MiniGridEnv,
    agent: str | Agent | type[Agent] = "ball",
    observation: str | ObservationSpecification = "no-arena",
    xy_scale: float = 1,
    timesteps: int = 500,
    task_function: callable | None = None,
    get_task_function: callable | None = None,
    **env_kwargs: dict,
) -> gym.Env:
    """Get a gymnasium environment for a minigrid gridworld with an agent instantiated."""
    # Imports are wrapped inside to avoid circular gymnasium imports.
    import gymnasium as gym
    from gymnasium.wrappers.env_checker import PassiveEnvChecker
    from gymnasium.wrappers.order_enforcing import OrderEnforcing

    from cocogrid.common.observation import get_observation_spec
    from cocogrid.tasks import TaskRegistry
    if isinstance(minigrid, gym.Env):
        minigrid_env = minigrid
    elif isinstance(minigrid, str):
        minigrid_env = gym.make(minigrid, disable_env_checker=True)
        if isinstance(minigrid_env, OrderEnforcing):
            minigrid_env = minigrid_env.env
        if isinstance(minigrid_env, PassiveEnvChecker):
            minigrid_env = minigrid_env.env

    if isinstance(agent, str):
        if not AgentRegistry.exists(agent):
            raise ValueError(f"{agent} is not a registered agent")
        agent = AgentRegistry.get(agent)
    elif isinstance(agent, Agent):
        agent = agent
    elif isinstance(agent, type) and issubclass(agent, Agent):
        agent = agent()
    else:
        raise TypeError("walker_type should be a string, Agent, or Agent type.")

    if task_function is not None:
        # the task function is the same every episode
        def get_task_function(minigrid: gym.Env) -> tuple[callable, str]:
            return (task_function, "")
    elif get_task_function is None:
        get_task_function = TaskRegistry.get_instance().get_task_for_env(type(minigrid_env))

    observation_spec = get_observation_spec(observation) if isinstance(observation, str) else observation
    engine = agent.get_engine()
    return engine.build_gymnasium_env(
        minigrid_env, agent, get_task_function, observation_spec, timesteps=timesteps, xy_scale=xy_scale, **env_kwargs)

IS_INITIALIZED = False
REGISTERED_GYM_IDS = set()


def register_environments_and_tasks() -> None:
    """Register all CocoGrid and custom Minigrid environments in gymnasium."""
    from gymnasium.envs.registration import register as gym_register
    from gymnasium.envs.registration import registry as gym_registry

    from cocogrid.minigrid import (
        register_base_minigrid_tasks,
        register_custom_minigrid_envs,
        register_custom_minigrid_tasks,
    )

    global IS_INITIALIZED
    IS_INITIALIZED = True

    # Ensure all Minigrid environments are registered to gym
    minigrid_env_ids = {env_spec.id for env_spec in gym_registry.values() if env_spec.id.startswith("MiniGrid")}
    minigrid_env_ids.update(register_custom_minigrid_envs())

    # Populate the TaskRegistry
    register_base_minigrid_tasks()
    register_custom_minigrid_tasks()

    # Register CocoGrid-equivalent environments.
    for minigrid_id in minigrid_env_ids:
        cocogrid_id = minigrid_id.replace("MiniGrid", "Cocogrid")
        gym_register(id=cocogrid_id, entry_point=functools.partial(get_cocogrid_gymnasium_env, minigrid_id))
        REGISTERED_GYM_IDS.add(cocogrid_id)
