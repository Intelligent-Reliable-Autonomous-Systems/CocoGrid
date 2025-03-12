"""Define the agent base class and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from cocogrid.common.registry import AlreadyRegisteredError, NotRegisteredError

if TYPE_CHECKING:
    from cocogrid.engine import Engine


class Agent(ABC):
    """The Agent base class specifies common properties of an agent, such as physics engine to use."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get the name of the agent."""
        pass

    @classmethod
    @abstractmethod
    def get_engine(cls) -> Engine:
        """Get the physics engine to use for the agent."""
        pass

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the agent with optional kwargs."""


class AgentRegistry:
    """A static class to register agent classes."""

    _agents: ClassVar[dict[str, type[Agent]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register(cls, agent_class: type[Agent], agent_id: str | None = None) -> None:
        """Register an agent class by its name or by a specified id."""
        agent_id = agent_id or agent_class.get_name()
        if agent_id in cls._agents:
            raise AlreadyRegisteredError(cls.__name__, agent_id)
        cls._agents[agent_id] = agent_class

    @classmethod
    def exists(cls, agent_id: str) -> bool:
        """Return whether agent_id is a registered agent id."""
        return agent_id in cls._agents

    @classmethod
    def get(cls, agent_id: str) -> type[Agent]:
        """Retrieve an agent class by its ID."""
        if agent_id not in cls._agents:
            raise NotRegisteredError(cls.__name__, agent_id)
        return cls._agents[agent_id]

    @classmethod
    def initialize(cls) -> None:
        """Mark as initialized."""
        cls._initialized = True

    @classmethod
    def is_initialized(cls) -> bool:
        """Return whether registry has been initialized."""
        return cls._initialized


def register_agents() -> None:
    """Register all the available agents into the registry."""
    from cocogrid.box2d.box2d_agent import Box2DAgent
    from cocogrid.mujoco.mujoco_agent import MuJoCoBallAgent

    AgentRegistry.initialize()
    AgentRegistry.register(MuJoCoBallAgent)
    AgentRegistry.register(Box2DAgent)
