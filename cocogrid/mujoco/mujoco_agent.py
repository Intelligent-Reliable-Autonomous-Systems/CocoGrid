from __future__ import annotations

from typing import TYPE_CHECKING

from cocogrid.agent import Agent
from cocogrid.mujoco.mujoco_engine import MujocoEngine
from cocogrid.mujoco.walkers.rolling_ball import RollingBallWithHead

if TYPE_CHECKING:
    import numpy as np
    from dm_control.locomotion.walkers.base import Walker
    from dm_control.mjcf.physics import Physics

    from cocogrid.engine import Engine
    from cocogrid.mujoco.cocogrid_arena import CocogridArena


class MuJoCoAgent(Agent):
    """Represent an Agent to be instantiated in the MuJoCo engine."""

    def __init__(self, arena: CocogridArena) -> None:
        """Construct a MuJoCoAgent."""
        self.arena = arena
        freejoints = [joint for joint in arena.mjcf_model.find_all("joint") if joint.tag == "freejoint"]
        self.freejoint = freejoints[0] if len(freejoints) > 0 else None
        self._walker = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Get the physics engine to use for the agent."""
        return MujocoEngine()

    @property
    def walker(self) -> Walker:
        """Get the locomotion Walker model."""
        return self._walker

    def get_walker_pos(self, physics: Physics) -> np.ndarray:
        """Get the walker's position."""
        if self.freejoint is not None:
            return physics.bind(self.freejoints[0]).qpos
        return physics.bind(self.walker.root_body).xpos

    def get_walker_vel(self, physics: Physics) -> np.ndarray:
        """Get the walker's velocity."""
        if self.freejoint is not None:
            return physics.bind(self.freejoints[0]).qvel
        return physics.bind(self.walker.root_body).cvel[:3]


class MuJoCoBallAgent(MuJoCoAgent):
    """Represent the dm_control.locomotion RollingBallWithHead entity."""

    def __init__(self, arena: CocogridArena) -> None:
        """Construct a MuJoCoBallAgent."""
        super().__init__(arena)
        self._walker = RollingBallWithHead(initializer=())

    @classmethod
    def get_name(cls) -> str:
        """Get the agent name."""
        return "ball"
