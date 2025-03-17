from __future__ import annotations

import hashlib
import pickle
import re
import sys
from collections import deque
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from cocogrid.common.abstraction.goal_wrapper import (
    DeterministicValueIterationPlanner,
    DjikstraBackwardsPlanner,
    GoalObserver,
    GoalWrapper,
)
from cocogrid.common.cocogrid_state import CocogridState
from cocogrid.common.color import COLOR_MAP, get_color_idx
from cocogrid.common.entity import ObjectEnum


class GridAbstraction:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    ACTION_GRAB = 4
    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_GRAB]
    OBJECT_IDS = ["ball", "box", "door", "key"]
    DOOR_IDX = ObjectEnum.DOOR.value
    KEY_IDX = ObjectEnum.KEY.value
    GRID_EMPTY = 0
    GRID_WALL = 1
    GRID_GOAL = 2
    GRID_LAVA = 3
    DOOR_STATE_CLOSED = 0
    DOOR_STATE_OPEN = 1

    def __init__(
        self, grid: np.ndarray, walker_pos: Tuple[int], objects: List[Tuple[int]], snap_held_to_agent=True
    ) -> None:
        self.grid: np.ndarray = grid
        self._grid_hash: str = hash_ndarray(self.grid)

        self.walker_pos = walker_pos

        self.objects = objects.copy()
        self._doors = [obj for obj in self.objects if obj[0] == GridAbstraction.DOOR_IDX]
        self._held_object = next(
            (i for i, o in enumerate(self.objects) if o[0] != GridAbstraction.DOOR_IDX and o[4] > 0), -1
        )
        if snap_held_to_agent and self._held_object > -1:
            # held object should be snapped to walker position
            idx, _, _, color, state = self.objects[self._held_object]
            self.objects[self._held_object] = (idx, walker_pos[0], walker_pos[1], color, state)
        self._objects_hash = hash_list_of_tuples(self.objects)

        # solver = GridSolver()
        # solver.solve_state(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridAbstraction):
            return False
        return (
            self.walker_pos == other.walker_pos
            and self._grid_hash == other._grid_hash
            and self._objects_hash == other._objects_hash
        )

    def __hash__(self) -> int:
        return hash((*self.walker_pos, self._grid_hash, self._objects_hash))

    def __repr__(self) -> str:
        obj_str = ",".join(map(GridAbstraction.pretty_object, self.objects))
        return f"Grid[{self.walker_pos}; {obj_str}]"

    def do_action(self, action: int) -> GridAbstraction:
        new_objects = self.objects.copy()
        held_object = self._held_object
        new_pos = self.walker_pos

        if action < GridAbstraction.ACTION_GRAB:
            offsets = [(0, -1), (-1, 0), (0, 1), (1, 0)]
            off_x, off_y = offsets[action]
            new_pos = self.walker_pos[0] + off_x, self.walker_pos[1] + off_y
            if self.grid[new_pos] == GridAbstraction.GRID_WALL:
                return self

            for idx, (type, col, row, color, state) in enumerate(self.objects):
                if type != GridAbstraction.DOOR_IDX:
                    continue
                if (col, row) == new_pos:  # agent is at door
                    # breakpoint()
                    if state > 1:  # is locked
                        if self._held_object == -1:
                            return self
                        else:
                            held_type, _, _, held_color, _ = self.objects[self._held_object]
                            if held_type != GridAbstraction.KEY_IDX or color != held_color:  # cannot unlock
                                return self
                    new_objects[idx] = (type, col, row, color, 0)  # open door
        else:
            index_to_grab = action - GridAbstraction.ACTION_GRAB
            if held_object == -1:
                # not holding anything.
                holdable_objects = [
                    idx
                    for idx, (id, col, row, _, _) in enumerate(self.objects)
                    if id != GridAbstraction.DOOR_IDX and (col, row) == self.walker_pos
                ]
                if index_to_grab >= len(holdable_objects):
                    return self
                held_object = holdable_objects[index_to_grab]
            elif index_to_grab > 0:
                return self
            else:
                new_objects[held_object] = (*new_objects[held_object][:4], 0)
                held_object = -1  # let go of the object

        if held_object != -1:
            idx, col, row, color, _ = new_objects[held_object]
            new_objects[held_object] = (idx, *new_pos, color, 1)
        return GridAbstraction(self.grid, new_pos, new_objects)

    def get_neighbors(self):
        neighbors = set([self.do_action(action) for action in GridAbstraction.ACTIONS])
        extra_grab_action = GridAbstraction.ACTION_GRAB + 1
        next_grab_state = None
        # multiple objects can be in the same position. so keep doing grab actions until there are no more
        while next_grab_state is not self:
            next_grab_state = self.do_action(extra_grab_action)
            neighbors.add(next_grab_state)
            extra_grab_action += 1
        if self in neighbors:
            neighbors.remove(self)
        return neighbors

    @property
    def walker_grid_cell(self):
        # print(self.walker_pos)
        return self.grid[self.walker_pos]

    @property
    def grid_width(self):
        return self.grid.shape[0]

    @property
    def grid_height(self):
        return self.grid.shape[1]

    @property
    def held_object_idx(self) -> int:
        """Get the index of a held object or -1 if none held."""
        return self._held_object

    @staticmethod
    def grid_distance_from_state(grid_pos, cocogrid_state: CocogridState):
        wx = cocogrid_state.pose[0] / cocogrid_state.xy_scale
        wy = -cocogrid_state.pose[1] / cocogrid_state.xy_scale

        def clamp(x, a, b):
            return max(a, min(x, b))

        tx = clamp(wx, grid_pos[0], grid_pos[0] + 1)
        ty = clamp(wy, grid_pos[1], grid_pos[1] + 1)

        return (wx - tx) ** 2 + (wy - ty) ** 2

    @staticmethod
    def distance_between_states(
        cocogrid_state: CocogridState, cur_abstract: GridAbstraction, target_abstract: GridAbstraction
    ):
        total_dist = 0
        wx = cocogrid_state.pose[0] / cocogrid_state.xy_scale
        wy = -cocogrid_state.pose[1] / cocogrid_state.xy_scale

        if cur_abstract.walker_pos != target_abstract.walker_pos:

            def clamp(x, a, b):
                return max(a, min(x, b))

            grid_pos = target_abstract.walker_pos
            tx = clamp(wx, grid_pos[0], grid_pos[0] + 1)
            ty = clamp(wy, grid_pos[1], grid_pos[1] + 1)
            total_dist += (wx - tx) ** 2 + (wy - ty) ** 2

        for obj_state, cur_obj, targ_obj in zip(cocogrid_state.objects, cur_abstract.objects, target_abstract.objects):
            if cur_obj[0] == GridAbstraction.DOOR_IDX:
                # door shaping
                if targ_obj[4] == 0 and cur_obj[4] > 0:
                    # door needs to be opened. distance based on angle from open
                    total_dist = min((np.pi / 4) - abs(obj_state[4]), 1) / 2
                    # also reward going towards door
                    ox, oy = (
                        obj_state[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                        / cocogrid_state.xy_scale
                    )
                    oy *= -1
                    if cur_obj[4] >= 2:  # locked
                        door_color = cur_obj[3]
                        for targ_state, targ_obj in zip(cocogrid_state.objects, cur_abstract.objects):
                            if targ_obj[3] == door_color and targ_obj[0] == GridAbstraction.KEY_IDX and targ_obj[4] > 0:
                                # is a held key that can unlock door
                                kx, ky = (
                                    targ_state[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                                    / cocogrid_state.xy_scale
                                )
                                ky = -ky
                                # total_dist += ((kx - ox)**2 + (ky - oy)**2) / 2.48
                                total_dist += np.linalg.norm([kx - ox, ky - oy]) / 3
                                break
                    else:
                        # total_dist += ((wx - ox)**2 + (wy - oy)**2) / 2.48
                        total_dist += np.linalg.norm([wx - ox, wy - oy]) / 3
                    break
            else:
                # check held object
                if targ_obj[4] == 1 and cur_obj[4] == 0:
                    # need to hold object; go towards it
                    ox, oy = (
                        obj_state[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                        / cocogrid_state.xy_scale
                    )
                    oy *= -1
                    total_dist += ((wx - ox) ** 2 + (wy - oy) ** 2) / 1.41

        return total_dist

    # @staticmethod
    # def distance_vel_from_state(cocogrid_state: CocogridState, cur_abstract: GridAbstraction, target_abstract: GridAbstraction):
    #     """Compute a continuous 'distance' from the target abstract state based on distance and velocity"""
    #     # get agent position and velocity
    #     from_pos = cocogrid_state.pose[CocogridState.POSE_IDX_POS:CocogridState.POSE_IDX_POS+2] / cocogrid_state.xy_scale
    #     vel = cocogrid_state.pose[CocogridState.POSE_IDX_VEL:CocogridState.POSE_IDX_VEL+2]

    #     dist_weight, vel_weight = 0.5, 0.5
    #     targ_pos = None
    #     total = 0

    #     if cur_abstract.walker_pos != target_abstract.walker_pos:
    #         # agent needs to move to the boundary of the next subogal
    #         tx, ty = target_abstract.walker_pos
    #         targ_pos = np.clip(from_pos, [tx, -(ty+1)], [tx+1, -ty])

    #     for obj_state, cur_obj, targ_obj in zip(cocogrid_state.objects, cur_abstract.objects, target_abstract.objects):
    #         if cur_obj[0] == GridAbstraction.DOOR_IDX:
    #             # door shaping
    #             if targ_obj[4] == 0 and cur_obj[4] > 0:
    #                 # door needs to be opened. distance based on angle from open
    #                 total = min(np.pi / 4 - abs(obj_state[4]), 1) * 0.4
    #                 # weight the velocity less to not punish going against the door
    #                 dist_weight, vel_weight = 0.5, 0.1
    #                 # set the door as the target position
    #                 targ_pos = obj_state[CocogridState.OBJECT_IDX_POS:CocogridState.OBJECT_IDX_POS+2] / cocogrid_state.xy_scale
    #                 if cur_obj[4] >= 2: # locked
    #                     door_color = cur_obj[3]
    #                     for targ_state, targ_obj in zip(cocogrid_state.objects, cur_abstract.objects):
    #                         if targ_obj[3] == door_color and targ_obj[0] == GridAbstraction.KEY_IDX and targ_obj[4] > 0:
    #                             # is a held key that can unlock door. use the key as from_pos to encourage it to go to the door
    #                             from_pos = targ_state[CocogridState.OBJECT_IDX_POS:CocogridState.OBJECT_IDX_POS+2] / cocogrid_state.xy_scale
    #                             vel = targ_state[CocogridState.OBJECT_IDX_VEL:CocogridState.OBJECT_IDX_VEL+2]
    #                             break
    #         else:
    #             # check held object
    #             if targ_obj[4] == 1 and cur_obj[4] == 0:
    #                 # need to hold object; set it as the target position
    #                 targ_pos = obj_state[CocogridState.OBJECT_IDX_POS:CocogridState.OBJECT_IDX_POS+2] / cocogrid_state.xy_scale

    #     if targ_pos is not None:
    #         dist = np.linalg.norm(targ_pos - from_pos)
    #         direction = (targ_pos - from_pos) / dist
    #         aligned_vel = np.dot(vel, direction)

    #         # distance should be near 1 when far away and near 0 when close
    #         dist_part = 1 - np.exp(-2 * dist)
    #         # velocity should be near 1 when not moving towards target and near 0 when moving fast towards it
    #         vel_part = min(1, np.exp(-0.9 * aligned_vel))
    #         print('distvel', vel, aligned_vel, dist_part, vel_part)
    #         total += dist_weight * dist_part + vel_weight * vel_part
    #     return total

    @staticmethod
    def from_cocogrid_state(cocogrid_state: CocogridState, force_door_evict=False, snap_held_to_agent=True):
        def obj_to_grid(object_state: np.ndarray):
            id = object_state[0]
            col, row = GridAbstraction.continuous_position_to_grid(object_state[1:4], cocogrid_state.xy_scale)
            color = object_state[14]
            state = object_state[15]
            return int(id), col, row, int(color), int(state)

        objects = [obj_to_grid(obj) for obj in cocogrid_state.objects]
        walker_pos = GridAbstraction.continuous_position_to_grid(cocogrid_state.pose[:3], cocogrid_state.xy_scale)
        for obj, abstract in zip(cocogrid_state.objects, objects):
            if abstract[0] == GridAbstraction.DOOR_IDX and (abstract[4] > 0 or force_door_evict):
                if abstract[1:3] == walker_pos:
                    # if the agent is in the same spot as a closed door, move them out of that square
                    pos_diff = np.sign(
                        cocogrid_state.pose[:2] - obj[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                    ).astype(int)
                    pos_diff[1] *= -1
                    door_dir = obj[CocogridState.OBJECT_IDX_DOOR_ORIENTATION] * np.pi
                    # either mask out the x or y offset
                    pos_offset = pos_diff * ((1, 0) if abs(np.cos(door_dir)) > abs(np.sin(door_dir)) else (0, 1))
                    walker_pos = (walker_pos[0] + pos_offset[0], walker_pos[1] + pos_offset[1])
                for idx, (other_obj, other_abstract) in enumerate(zip(cocogrid_state.objects, objects)):
                    if other_abstract[0] != GridAbstraction.DOOR_IDX and abstract[1:3] == other_abstract[1:3]:
                        # other object should be pushed out of door
                        pos_diff = np.sign(
                            other_obj[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                            - obj[CocogridState.OBJECT_IDX_POS : CocogridState.OBJECT_IDX_POS + 2]
                        ).astype(int)
                        pos_diff[1] *= -1
                        door_dir = obj[CocogridState.OBJECT_IDX_DOOR_ORIENTATION] * np.pi
                        # either mask out the x or y offset
                        new_pos = (
                            pos_diff * ((1, 0) if abs(np.cos(door_dir)) > abs(np.sin(door_dir)) else (0, 1))
                            + other_abstract[1:3]
                        )
                        objects[idx] = (other_abstract[0], *new_pos, *other_abstract[3:5])

        return GridAbstraction(cocogrid_state.grid, walker_pos, objects, snap_held_to_agent=snap_held_to_agent)

    @staticmethod
    def continuous_position_to_grid(pos, xy_scale):
        col = int(np.floor(pos[0] / xy_scale))
        row = int(np.floor(-pos[1] / xy_scale))
        return col, row

    @staticmethod
    def pretty_object(object_tuple: Tuple[int]):
        oid, col, row, color_id, state = object_tuple
        name = "unknown"
        if 0 <= oid < len(GridAbstraction.OBJECT_IDS):
            name = GridAbstraction.OBJECT_IDS[oid]
        color = list(COLOR_MAP.keys())[color_id]
        return f"[{color} {name} at ({col},{row}): {state}]"

    @staticmethod
    def backward_neighbor_edges(state: GridAbstraction):
        # we assume that actions are bidirectional. This should be the case, except maybe when objects are in the same cell
        neighbors = state.get_neighbors()
        return [(neighbor, 1) for neighbor in neighbors]


def get_cocogrid_goal_wrapper(env: gym.Env, env_id: str, cls=GoalWrapper):
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_cocogrid_state(state)

    planner = DjikstraBackwardsPlanner(GridAbstraction.backward_neighbor_edges)
    if "RandomObject" in env_id:

        def goal_fn(obs, abstract, _env):
            task = _env.unwrapped._task
            pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
            matches = re.search(pattern, task.description)

            if not matches:
                raise Exception(f"Task '{task}' does not meet specification for RandomObject")
            color = matches.group(1)
            color_idx = get_color_idx(color)
            class_name = matches.group(2)
            class_idx = ["Ball", "Box"].index(class_name)
            x = int(matches.group(3))
            y = int(matches.group(4))
            objects = abstract.objects.copy()
            for idx, obj in enumerate(abstract.objects):
                if obj[0] == class_idx and obj[3] == color_idx:
                    objects[idx] = (obj[0], x, y, obj[3], 0)

            return GridAbstraction(abstract.grid, (x, y), objects)

        def goal_obs_fn(abstract):
            obj_pos = abstract.objects[abstract._held_object][1:3] if abstract._held_object >= 0 else (-1, -1)
            return (*abstract.walker_pos, *obj_pos, abstract._held_object)

        low = [0, 0, 0, 0, 0]
        high = [5, 5, 5, 5, 5]
    else:

        def goal_fn(obs, abstract, _env):
            goal_idx = np.where(abstract.grid == GridAbstraction.GRID_GOAL)
            goal_x, goal_y = goal_idx[0][0], goal_idx[1][0]
            return GridAbstraction(abstract.grid, (goal_x, goal_y), abstract.objects)

        goal_obs_fn = lambda abstract: abstract.walker_pos
        low = [0, 0]
        high = [np.inf, np.inf]

    observer = GoalObserver(goal_obs_fn, low, high)
    return cls(env, abstraction_fn, goal_fn, planner, observer)


def get_cocogrid_goal_wrapper_2(env: gym.Env, env_id: str, cls=GoalWrapper):
    def abstraction_fn(obs, _env):
        state = _env.unwrapped.state
        return GridAbstraction.from_cocogrid_state(state)

    planner = DeterministicValueIterationPlanner(GridAbstraction.ACTIONS, lambda state, action: state.do_action(action))
    if "RandomObject" in env_id:

        def task_fn(obs, abstract, _env):
            """This returns a function that checks if the specified object is delivered to the goal"""
            task = _env.unwrapped._task
            pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
            matches = re.search(pattern, task.description)

            if not matches:
                raise Exception(f"Task '{task}' does not meet specification for RandomObject")
            color = matches.group(1)
            color_idx = get_color_idx(color)
            class_name = matches.group(2)
            class_idx = ["Ball", "Box"].index(class_name)
            x = int(matches.group(3))
            y = int(matches.group(4))
            target = (class_idx, x, y, color_idx, 0)

            def task_status(abstract):
                if abstract.walker_grid_cell == GridAbstraction.GRID_LAVA:
                    return -1, True
                for obj in abstract.objects:
                    if obj == target:
                        return 1, True
                return 0, False

            return task_status

        def goal_obs_fn(abstract):
            obj_pos = abstract.objects[abstract._held_object][1:3] if abstract._held_object >= 0 else (-1, -1)
            return (*abstract.walker_pos, *obj_pos, abstract._held_object)

        low = [0, 0, 0, 0, 0]
        high = [5, 5, 5, 5, 5]
    else:

        def task_fn(obs, abstract, _env):
            def test_status(abstract):
                if abstract.walker_grid_cell == GridAbstraction.GRID_LAVA:
                    return -1, True
                if abstract.walker_grid_cell == GridAbstraction.GRID_GOAL:
                    return 1, True
                return 0, False

            return test_status

        goal_obs_fn = lambda abstract: abstract.walker_pos
        low = [0, 0]
        high = [np.inf, np.inf]

    observer = GoalObserver(goal_obs_fn, low, high)
    return cls(env, abstraction_fn, task_fn, planner, observer)


class GridSolver:
    def __init__(self):
        self.grid_state = None

    def solve_state(self, grid_state: GridAbstraction):
        self.grid_state = grid_state
        self.component_connections = self.get_all_connections()
        # print(self.component_connections)
        # print(list(map(lambda x: GridAbstraction.pretty_object(x), self.component_connections.keys())))

    def get_all_connections(self):
        self._agent = (-1, *self.grid_state.walker_pos, 0, 0)
        self._components = [self._agent, *self.grid_state.objects]
        # objects = []
        # for i in range(len(self.minigrid.grid.grid)):
        #     r, c = self.get_row_col(i)
        #     for world_obj in self.get_world_objects(i):
        #         if type(world_obj) in MinigridSolver.object_types:
        #             # c, r = world_obj.cur_pos
        #             world_obj.cur_pos = c, r
        #             objects.append((self.get_flat_idx(r, c), world_obj))
        all_connections = {component: self.connected_components(component[1:3]) for component in self._components}
        return all_connections

    def connected_components(self, start_pos):
        """Find connections to all components from starting position"""

        to_visit = deque([(start_pos, 0)])
        expanded = set([start_pos])
        grid_to_from = dict()
        grid_to_from[start_pos] = start_pos
        connected = []

        while len(to_visit) > 0:
            search_pos, dist = to_visit.popleft()
            grid_dir = self.get_grid_dir(grid_to_from[search_pos], search_pos)
            neighbors = self.get_neighbors(search_pos, dir=grid_dir)
            for neighbor_pos in neighbors:
                if neighbor_pos in expanded:
                    continue
                expanded.add(neighbor_pos)
                grid_to_from[neighbor_pos] = search_pos

                is_blocking = False
                for component in self._components:
                    if component[1:3] == neighbor_pos:
                        path_back = self.get_path_back(grid_to_from, start_pos, neighbor_pos)
                        connected.append((component, dist + 1, path_back))
                        if (
                            component[0] == GridAbstraction.DOOR_IDX
                            and component[4] == GridAbstraction.DOOR_STATE_CLOSED
                        ):
                            is_blocking = True

                cell_value = self.grid_state.grid[neighbor_pos]
                if cell_value == GridAbstraction.GRID_WALL or cell_value == GridAbstraction.GRID_LAVA:
                    # not blocking
                    is_blocking = True

                if not is_blocking:
                    to_visit.append((neighbor_pos, dist + 1))
        return connected

    def get_grid_dir(self, from_pos, to_pos):
        diff = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        if diff[0] >= 1:
            # then to_idx is to the right
            return 0
        if diff[1] >= 1:
            # to_idx is below
            return 1
        if diff[0] <= -1:
            # to_idx is to the left
            return 2
        # to_idx is above
        return 3

    def get_neighbors(self, pos, dir=0):
        width = self.grid_state.grid_width
        col, row = pos
        neighbors = [None, None, None, None]
        if col + 1 < width:
            # right
            neighbors[0] = (col + 1, row)
        if row + 1 < self.grid_state.grid_height:
            # down
            neighbors[1] = (col, row + 1)
        if col - 1 >= 0:
            # left
            neighbors[2] = (col - 1, row)
        if row - 1 >= 0:
            # up
            neighbors[3] = (col, row - 1)
        # prioritize dir, then adjacent, and lastly the opposite dir
        prioritized = [neighbors[dir], neighbors[dir - 3], neighbors[dir - 1], neighbors[dir - 2]]
        return [p for p in prioritized if p is not None]

    def get_path_back(self, grid_to_from, start_pos, end_pos):
        pos = end_pos
        path = [end_pos]
        while pos != start_pos:
            pos = grid_to_from[pos]
            path.append(pos)
            if len(path) > 1000:
                print("Path back has cycle")
                exit()
        return path


md5_kwargs = {}
if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    md5_kwargs = {"usedforsecurity": False}


# hashing utils
def hash_ndarray(arr):
    """Compute an MD5 hash for a NumPy array."""
    arr_bytes = arr.tobytes()
    hash_obj = hashlib.md5(arr_bytes, **md5_kwargs)
    return hash_obj.hexdigest()


def hash_list_of_tuples(list_of_tuples):
    """Compute a hash for a list of tuples."""
    list_bytes = pickle.dumps(list_of_tuples)
    hash_obj = hashlib.md5(list_bytes, **md5_kwargs)
    return hash_obj.hexdigest()
