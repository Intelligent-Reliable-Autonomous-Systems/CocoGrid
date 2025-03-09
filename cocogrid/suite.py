from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.envs.registration import register as gym_register
from gymnasium.envs.registration import registry as gym_registry
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from gymnasium.wrappers.order_enforcing import OrderEnforcing

from cocogrid.agent import Agent, AgentRegistry
from cocogrid.custom_minigrid import register_custom_minigrid
from cocogrid.multitask import register_multitask_minigrid
from cocogrid.state.tasks import DEFAULT_TASK_REGISTRY, get_null_task
from cocogrid.state import get_observation_spec, ObservationSpecification

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv


def get_cocogrid_gymnasium_env(
    minigrid: str | MiniGridEnv,
    walker_type: str | Agent | type[Agent] = "ball",
    observation_type: str | ObservationSpecification = "no-arena",
    task_function: callable | None = None,
    get_task_function: callable | None = None,
    **env_kwargs: dict,
) -> gym.Env:
    """Get a gymnasium environment for a minigrid gridworld with an agent instantiated."""
    if isinstance(minigrid, gym.Env):
        minigrid_env = minigrid
    elif isinstance(minigrid, str):
        minigrid_env = gym.make(minigrid, disable_env_checker=True)
        if isinstance(minigrid, PassiveEnvChecker):
            minigrid_env = minigrid_env.env
        if isinstance(minigrid, OrderEnforcing):
            minigrid_env = minigrid_env.env

    if isinstance(walker_type, str):
        if not AgentRegistry.exists(walker_type):
            raise ValueError(f"{walker_type} is not a registered agent")
        agent = AgentRegistry.get(walker_type)
    elif isinstance(walker_type, (type[Agent], Agent)):
        agent = walker_type
    else:
        raise TypeError("walker_type should be a string, Agent, or Agent type.")

    if task_function is not None:
        # the task function is the same every episode
        def get_task_function(minigrid: gym.Env) -> tuple[callable, str]:
            return (task_function, "")
    elif get_task_function is None:
        get_task_function = DEFAULT_TASK_REGISTRY.get(type(minigrid_env), get_null_task)

    observation_spec = get_observation_spec(observation_type) if isinstance(observation_type, str) else observation_type

    engine = agent.get_engine()
    return engine.build_gymnasium_env(minigrid_env, agent, get_task_function, observation_spec, **env_kwargs)


REGISTERED_GYM_IDS = set()


def register_environments_and_tasks() -> None:
    """Register all CocoGrid and custom Minigrid environments in gymnasium."""
    minigrid_env_ids = {env_spec.id for env_spec in gym_registry.values() if env_spec.id.startswith("MiniGrid")}
    minigrid_env_ids.update(register_custom_minigrid())
    minigrid_env_ids.update(register_multitask_minigrid())

    for minigrid_id in minigrid_env_ids:
        cocogrid_id = minigrid_id.replace("MiniGrid", "Cocogrid")
        gym_register(id=cocogrid_id, entry_point=functools.partial(get_cocogrid_gymnasium_env, minigrid_id))
        REGISTERED_GYM_IDS.add(cocogrid_id)
