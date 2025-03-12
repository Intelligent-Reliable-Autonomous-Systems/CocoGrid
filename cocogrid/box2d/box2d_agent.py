from __future__ import annotations

from typing import TYPE_CHECKING

from cocogrid.agent import Agent
from cocogrid.box2d.box2d_engine import Box2DEngine

if TYPE_CHECKING:
    from cocogrid.engine import Engine


class Box2DAgent(Agent):
    """Represent an Agent to be instantiated in the Box2D engine."""

    def __init__(self) -> None:
        """Construct a Box2DAgent."""
        self._body = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Get the physics engine to use for the agent."""
        return Box2DEngine()

    @classmethod
    def get_name(cls) -> str:
        """Get the agent name."""
        return "box2d"

    @property
    def body(self):
        """Get the Box2D model."""
        return self._body

    def construct_body(self, world, position):
        self._body = world.CreateDynamicBody(position=position)
        # circle = self.agent.CreateCircleFixture(radius=0.4, density=1, friction=0.3)
        self._body.CreatePolygonFixture(box=(0.3, 0.3), density=1, friction=0.3)
        self._body.linearDamping = 3
        self._body.fixedRotation = True
        self._color = (115, 115, 115, 255)

    def delete_body(self):
        del self._body
        self._body = None

    def draw_body(self, pixel_per_meter, height, surface):
        for fixture in self.body.fixtures:
            fixture.shape.draw(self.body, self._color, pixel_per_meter, height, surface)
