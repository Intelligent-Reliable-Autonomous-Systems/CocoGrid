"""Define the CocogridState dataclass, which represents the state of an environment."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from cocogrid.color import get_color_idx


@dataclass
class CocogridState:
    """Contain the engine-agnostic environment state.

    Instance Attributes:
    grid -- An integer matrix of static, grid-aligned elements, such as walls and lava.
    xy_scale -- A positive integer representing the width of one grid cell.
    objects -- An NxM matrix, where N is the number of objects in the scene
        and M is the dimension of each object's state.
    pose -- A vector representing the agent's position, orientation, velocity.
    walker -- A dictionary of other heterogenous features, such as proprioception or camera.

    Class Attributes:
    POSE_IDX_POS (int) -- The starting index of the agent's position in pose.
    POSE_IDX_QUAT (int) -- The starting index of the agent's orientation in pose.
    POSE_IDX_VEL (int) -- The starting index of the agent's velocity in pose.
    OBJECT_IDX_TYPE (int) -- The state index of an object's type identifier (e.g. 0 for box).
    OBJECT_IDX_POS (int) -- The starting state index of an object's position.
    OBJECT_IDX_DOOR_ORIENTATION -- The state index of a door object's orientation.
    OBJECT_IDX_VEL (int) -- The state index of an object's velocity.
    OBJECT_IDX_COLOR (int) -- The state index of an object's color identifier.
    OBJECT_IDX_STATE (int) -- The state index of an object's state attribute (e.g. 1 for held)
    """

    POSE_IDX_POS: ClassVar[int] = 0
    POSE_IDX_QUAT: ClassVar[int] = 3
    POSE_IDX_VEL: ClassVar[int] = 7
    OBJECT_IDX_TYPE: ClassVar[int] = 0
    OBJECT_IDX_POS: ClassVar[int] = 1
    OBJECT_IDX_DOOR_ORIENTATION: ClassVar[int] = 5
    OBJECT_IDX_VEL: ClassVar[int] = 8
    OBJECT_IDX_COLOR: ClassVar[int] = 14
    OBJECT_IDX_STATE: ClassVar[int] = 15

    @staticmethod
    def color_to_idx(color: str) -> int:
        """Get the index of a color from its name."""
        return get_color_idx(color)

    def __init__(
        self, grid: np.ndarray, xy_scale: float, objects: np.ndarray, pose: np.ndarray, walker: dict[str, np.ndarray]
    ) -> None:
        """Construct a CocogridState.

        Input:
        grid -- An integer matrix of static, grid-aligned elements, such as walls and lava.
        xy_scale -- A positive integer representing the width of one grid cell.
        objects -- An NxM matrix, where N is the number of objects in the scene
            and M is the dimension of each object's state.
        pose -- A vector representing the agent's position, orientation, velocity.
        walker -- A dictionary of other heterogenous features, such as proprioception or camera.

        Note on the coordinate system:
        - The origin is in the top left corner.
        - The x axis increases to the right.
        - The y axis *decreases* to the bottom, in the negative quadrant.
        - Cells are defined by the top left corner, not the center.
        - This means that a Minigrid cell (3,5) would correspond to the range
            x=[3*xy_scale, 4*xy_scale), y=(-6*xy_scale, -5*xy_scale].
        """
        self.grid = grid
        self.grid.flags.writeable = False
        self.xy_scale = xy_scale
        self.objects = objects
        self.objects.flags.writeable = False
        self.pose = pose
        self.pose.flags.writeable = False
        self.walker = walker
        for arr in self.walker.values():
            if arr.ndim > 0:
                arr.flags.writeable = False

    def get_arena_size(self) -> np.ndarray:
        """Get the total shape of the arena, grid.shape * xy_scale."""
        return np.array(self.grid.shape) * self.xy_scale

    def get_walker_position(self, dim: Literal[2, 3] = 3) -> np.ndarray:
        """Get the position of the agent.

        Input:
        dim -- Optionally specify the number of dimensions (default 3).
        """
        return self.pose[:dim]

    def get_normalized_walker_position(self, without_border: bool = False) -> np.ndarray:
        """Get the position of the agent, normalized to a range [0,1] x [0,1]. Make the y axis positive.

        Input:
        without_border -- Optionally, normalize within the interior of the arena border.
        """
        pos = self.pose[:3].copy()
        pos[1] *= -1
        pos[:2] = (pos[:2] / self.xy_scale - int(without_border)) / (self.grid.shape[0] - 2 * int(without_border))
        return pos

    def get_walker_velocity(self, dim: Literal[2, 3] = 3) -> np.ndarray:
        """Get the agent velocity.

        Input:
        dim -- Optionally specify the number of dimensions (default 3).
        """
        return self.pose[7 : 7 + dim]

    @staticmethod
    def get_object_pos_slice(dim: Literal[2, 3] = 3) -> slice:
        """Get a slice of the index range for an object's position."""
        return slice(CocogridState.OBJECT_IDX_POS, CocogridState.OBJECT_IDX_POS + dim)

    @staticmethod
    def get_grid_state_from_minigrid(minigrid_env: MiniGridEnv) -> np.ndarray:
        """Take a minigrid environment and convert the grid into a static state representation.

        Input:
        - minigrid_env (MiniGridEnv) -- a minigrid environment

        Output:
        - grid_state (np.ndarray) -- a matrix with same dimensions as minigrid_env.grid,
            where 0 is nothing, 1 is wall, 2 is goal, and 3 is lava
        """
        object_mapping = {Wall: 1, Goal: 2, Lava: 3}
        width, height = minigrid_env.grid.width, minigrid_env.grid.height
        grid_state = np.zeros(shape=(width, height), dtype=int)
        for i in range(width):
            for j in range(height):
                # i is the col, j is the row
                world_obj = minigrid_env.grid.get(i, j)
                grid_state[i, j] = object_mapping.get(type(world_obj), 0)

        return grid_state
