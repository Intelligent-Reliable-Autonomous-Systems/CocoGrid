from __future__ import annotations

from typing import TYPE_CHECKING

from cocogrid.box2d.gym import Box2DEnv
from cocogrid.engine import Engine

if TYPE_CHECKING:
    import gymnasium as gym

    from cocogrid.agent import Agent
    from cocogrid.common.observation import ObservationSpecification


class Box2DEngine(Engine):
    """Manage constructing Box2D physics environments."""

    @property
    def name(self) -> str:
        """Get the physics engine name."""
        return "Box2D"

    def build_gymnasium_env(
        self,
        minigrid: gym.Env,
        agent: Agent | type[Agent],
        get_task_function: callable,
        observation_spec: ObservationSpecification,
        render_mode: str = "rgb_array",
        render_width: int = 64,
        **env_kwargs: dict,
    ) -> gym.Env:
        """Construct a Box2D gymnasium environment."""
        if isinstance(agent, type):
            agent = agent()

        return Box2DEnv(
            minigrid,
            agent,
            get_task_function,
            observation_spec=observation_spec,
            **env_kwargs,
        )
