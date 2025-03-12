"""Define the engine base class and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

    from cocogrid.agent import Agent
    from cocogrid.common.observation import ObservationSpecification

ENGINE_REGISTRY: dict[str, Engine] = {}


class Engine(ABC):
    """The Engine base class specifies common properties of a physics engine."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the engine."""

    @abstractmethod
    def build_gymnasium_env(
            self, minigrid: gym.Env, agent: Agent | type[Agent], get_task_function: callable,
            observation_spec: ObservationSpecification, **env_kwargs: dict) -> gym.Env:
        """Build a gymnasium environment in this physics engine."""
