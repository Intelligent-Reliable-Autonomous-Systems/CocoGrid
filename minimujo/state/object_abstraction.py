from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, List, Tuple

from minimujo.color import COLOR_MAP
import numpy as np
from minimujo.entities import ObjectEnum, get_color_id
from minimujo.state.grid_abstraction import GridAbstraction, hash_list_of_tuples, hash_ndarray
from minimujo.state.minimujo_state import MinimujoState

@dataclass(frozen=True)
class ObjectAbstraction:

    # constants
    DOOR_IDX: ClassVar[int] = ObjectEnum.DOOR.value
    KEY_IDX: ClassVar[int] = ObjectEnum.KEY.value
    GOAL_IDX: ClassVar[int] = 4 # the index of the goal type
    OBJECT_IDS: ClassVar[List[int]] = ['ball', 'box', 'door', 'key', 'goal']

    # the type, color, and state of all objects in the scene
    objects: Tuple[Tuple[str, str, int]]

    # the rooms that all of the objects and agent are in
    # agent is index 0, object indices shifted up by 1 (e.g. objects[0] is index 1)
    rooms: Tuple[Tuple[int]]

    # which objects/agent are near to each other, using same index as rooms
    nears: Tuple[Tuple[int, int]]

    # a hacky way to keep track of positions for debugging
    _positions: List[Tuple[int, int]] = field(compare=False, default=None)

    def __post_init__(self):
        for room in self.rooms:
            if len(room) == 0:
                raise Exception(f"Cannot have empty room without door: {self}")

        def validate_door(door_idx):
            count_per_list = [lst.count(door_idx+1) for lst in self.rooms]
            distinct_lists_with_number = sum(1 for count in count_per_list if count > 0)
            total_occurrences = sum(count_per_list)
            return distinct_lists_with_number == 2 and total_occurrences == 2
        for idx, (obj_type, _, _) in enumerate(self.objects):
            if obj_type == ObjectAbstraction.DOOR_IDX:
                # each door must appear in exactly 2 distinct rooms
                if not validate_door(idx):
                    raise Exception(f"Door {idx} is not connected between exactly 2 rooms: {self}")
                
    def __repr__(self) -> str:
        obj_str = ','.join(map(ObjectAbstraction.pretty_object, self.objects))
        return f'Objects[{obj_str}]\n{self.nears}\n{self.rooms}'
    
    def go_near_object(self, target_idx):
        """The agent moves near an object in the same room"""

        agent_room = self.agent_room

        if not self.is_in_room(target_idx, agent_room):
            # cannot move to an object in a different room
            return self
        
        if self.is_holding(target_idx):
            # can't move to holding
            return self
        
        if self.are_near(0, target_idx+1):
            # already near target
            return self
        
        # move the agent and held objects
        # moved_indices = [0, *[idx+1 for idx, (obj_type, _, state) in enumerate(self.objects) if obj_type != ObjectAbstraction.DOOR_IDX and state >= 1]]

        # remove old nears
        filtered_pairs = tuple(
            pair for pair in self.nears
            if pair[0] != 0 and pair[1] != 0
        )

        # agent and held objects are near to 
        # added_pairs = tuple(
        #     tuple(sorted((idx, target_idx+1))) for idx in moved_indices
        # )

        new_nears = tuple(sorted(filtered_pairs + ((0, target_idx+1),)))

        return ObjectAbstraction(self.objects, self.rooms, new_nears, _positions=self._positions)

    def move_away_from_objects(self):
        # # move the agent and held objects
        # moved_indices = [0, *[idx+1 for idx, (obj_type, _, state) in enumerate(self.objects) if obj_type != ObjectAbstraction.DOOR_IDX and state >= 1]]

        # remove old nears
        new_nears = tuple(
            pair for pair in self.nears
            if pair[0] != 0 and pair[1] != 0
        )

        return ObjectAbstraction(self.objects, self.rooms, new_nears, _positions=self._positions)
    
    def pick_up_object(self, target_idx):
        if self.get_held_object() != -1:
            # already holding object
            return self
        
        if not self.can_hold(target_idx):
            # cannot pick up unholdable
            return self
        
        if not self.are_near(0, target_idx+1):
            # cannot pick up far object
            return self
        
        # update object state
        new_objects =  tuple(
            (value if i != target_idx else (*value[:2], 1))  # Set the object to held
            for i, value in enumerate(self.objects)
        )

        # remove all nears
        new_nears = tuple(
            pair for pair in self.nears
            if pair[0] != target_idx and pair[1] != target_idx
        )

        return ObjectAbstraction(new_objects, self.rooms, new_nears, _positions=self._positions)
    
    def drop_object(self):
        held_idx = self.get_held_object()

        if held_idx == -1:
            # not holding an object to drop
            return self
        
        # update object state
        new_objects =  tuple(
            (value if i != held_idx else (*value[:2], 0))  # Set the object to not held
            for i, value in enumerate(self.objects)
        )

        # get objects that are near to agent and agent itself
        near_agent = [0, *[idx_b for idx_a, idx_b in self.nears if idx_a == 0]]

        # add dropped object to nears
        new_nears = self.nears + tuple(
            tuple(sorted((idx, held_idx+1))) for idx in near_agent
        )
        new_nears = tuple(sorted(new_nears))

        return ObjectAbstraction(new_objects, self.rooms, new_nears, _positions=self._positions)
    
    def open_door(self):
        # get near door
        near_doors = [idx_b-1 for idx_a, idx_b in self.nears if idx_a == 0 and self.objects[idx_b-1][0] == ObjectAbstraction.DOOR_IDX]
        if len(near_doors) == 0:
            # not near a door
            return self
        
        near_door = near_doors[0]

        if not self.is_closed(near_door):
            # door is already open
            return self
        
        if self.is_locked(near_door):
            # agent must be holding key of same color
            held_idx = self.get_held_object()
            if held_idx == -1:
                # not holding anything
                return self
            held_obj = self.objects[held_idx]
            if held_obj[0] != ObjectAbstraction.KEY_IDX or held_obj[1] != self.objects[near_door][1]:
                # not a key of same color
                return self
            
        # update door state
        new_objects =  tuple(
            (value if i != near_door else (*value[:2], 0))  # Set the door to open and unlocked
            for i, value in enumerate(self.objects)
        )

        return ObjectAbstraction(new_objects, self.rooms, self.nears, _positions=self._positions)
    
    def go_through_door(self):
        # get near door
        near_doors = [idx_b-1 for idx_a, idx_b in self.nears if idx_a == 0 and self.objects[idx_b-1][0] == ObjectAbstraction.DOOR_IDX]
        if len(near_doors) == 0:
            # not near a door
            return self
        
        near_door = near_doors[0]

        if self.is_closed(near_door):
            # door is not open
            return self
        
        moved_indices = [0, *[idx+1 for idx, (obj_type, _, state) in enumerate(self.objects) if obj_type != ObjectAbstraction.DOOR_IDX and state >= 1]]
        agent_room = self.agent_room
        connecting_room = self.get_door_connecting_room(near_door, agent_room)

        # move away from agent_room to connecting_room
        new_rooms = tuple(
            (
                tuple(val for i, val in enumerate(self.rooms[agent_room]) if val not in moved_indices) if idx == agent_room else
                tuple(self.rooms[connecting_room] + tuple(moved_indices)) if idx == connecting_room else
                value  # Leave other tuples unchanged
            )
            for idx, value in enumerate(self.rooms)
        )

        return ObjectAbstraction(self.objects, new_rooms, self.nears, _positions=self._positions)
    
    def get_possible_actions(self):
        next_states = dict()

        try:
            agent_room = self.agent_room
        except:
            # not valid state
            return dict()
        for entity_idx in self.rooms[agent_room]:
            if entity_idx > 0:
                # go near object
                go_near = self.go_near_object(entity_idx-1)
                next_states[f'(go_near {entity_idx-1})'] = go_near

                # pick up object
                pick_up = self.pick_up_object(entity_idx-1)
                next_states[f'(pick_up {entity_idx-1})'] = pick_up

        # move away from object
        move_away = self.move_away_from_objects()
        next_states['(move_away_from_objects)'] = move_away

        # drop object
        drop_obj = self.drop_object()
        next_states['(drop_object)'] = drop_obj

        # open door
        open_door = self.open_door()
        next_states['(open_door)'] = open_door

        # go through door
        go_through = self.go_through_door()
        next_states['(go_through_door)'] = go_through

        return { action:next for action, next in next_states.items() if next != self}

    @property
    def agent_room(self):
        for idx, room in enumerate(self.rooms):
            if 0 in room:
                return idx
        raise Exception(f"Invalid abstract state: agent clipped into the backrooms: {self}")
    
    def is_in_room(self, object_idx, room_idx):
        return object_idx+1 in self.rooms[room_idx]
    
    def are_near(self, entity_a, entity_b):
        if entity_a > entity_b:
            entity_a, entity_b = entity_b, entity_a
        return (entity_a, entity_b) in self.nears
    
    def is_holding(self, object_idx):
        return self.objects[object_idx][2] >= 1 and self.can_hold(object_idx)
    
    def can_hold(self, object_idx):
        obj_type = self.objects[object_idx][0]
        return obj_type != ObjectAbstraction.DOOR_IDX and obj_type != ObjectAbstraction.GOAL_IDX
    
    def is_locked(self, door_idx):
        return (self.objects[door_idx][2] & 2) > 0
    
    def is_closed(self, door_idx):
        return (self.objects[door_idx][2] & 1) > 0
    
    def get_door_connecting_room(self, door_idx, current_room):
        for idx, room in enumerate(self.rooms):
            if idx != current_room and (door_idx+1) in room:
                return idx
        raise Exception(f"Door {door_idx} does not connect to anything from room {current_room}: {self}")
    
    def get_held_object(self):
        for idx, (_, _, state) in enumerate(self.objects):
            if self.can_hold(idx) and state >= 1:
                return idx
        return -1
    
    def get_near_object(self):
        for entity_a, entity_b in self.nears:
            if entity_a == 0:
                return entity_b - 1
        return -1

    @staticmethod
    def from_minimujo_state(state: MinimujoState):
        grid = GridAbstraction.from_minimujo_state(state, force_door_evict=True)

        # flood fill connections for same_room(A,B) and same_tile(A,B)
        rooms, positions = get_rooms_with_objects(grid)
        rooms = tuple(tuple(sorted(room)) for room in rooms)
        
        # object type, color, state
        objects = [(obj_type, color, state) for (obj_type, _, _, color, state) in grid.objects]
        objects += ((ObjectAbstraction.GOAL_IDX, get_color_id('green'), 0),) * (len(positions) - len(objects) - 1)
        objects = tuple(objects)

        # near(A,B)
        nears = []

        def is_tile_type(idx):
            # if is a goal or door
            return idx != 0 and objects[idx-1][0] == ObjectAbstraction.GOAL_IDX
        def get_position(idx):
            if idx == 0:
                # agent
                return state.get_walker_position()[:2]
            elif idx <= len(state.objects):
                # object
                return state.objects[idx-1,MinimujoState.OBJECT_IDX_POS:MinimujoState.OBJECT_IDX_POS+2]
            else:
                return None
        def is_held(idx):
            return idx != 0 and objects[idx-1][0] != ObjectAbstraction.DOOR_IDX and objects[idx-1][2] >= 1
        for idx_a in range(len(positions)):
            pos_a = get_position(idx_a)
            is_tile_a = is_tile_type(idx_a)
            if is_held(idx_a):
                continue
            for idx_b in range(idx_a+1, len(positions)):
                if is_tile_a or is_tile_type(idx_b):
                    if positions[idx_a] == positions[idx_b]:
                        nears.append((idx_a, idx_b))
                    continue
                elif is_held(idx_b):
                    # for convenience, a held object is never near
                    continue
                pos_b = get_position(idx_b)
                if np.linalg.norm(pos_a - pos_b) < 1:
                    # if objects are within 1 unit, they are near
                    nears.append((idx_a, idx_b))
        nears = tuple(nears)

        return ObjectAbstraction(objects, rooms, nears, _positions=positions)
    
    @staticmethod
    def set_adjacency(adjacencies, idx_a, idx_b, val):
        adjacencies[idx_a, idx_b] = val
        adjacencies[idx_b, idx_a] = val
    
    @staticmethod
    def pretty_object(object_tuple: Tuple[int]):
        oid, color_id, state = object_tuple
        name = 'unknown'
        if 0 <= oid < len(ObjectAbstraction.OBJECT_IDS):
            name = ObjectAbstraction.OBJECT_IDS[oid]
        color = list(COLOR_MAP.keys())[color_id]
        if oid == ObjectAbstraction.DOOR_IDX:
            status = ['open', 'closed', 'locked', 'locked'][state]
        elif oid == ObjectAbstraction.GOAL_IDX:
            status = ''
        else:
            status = ['floor', 'held'][state]
        return f'[{color} {name}: {status}]'

def get_rooms_with_objects(grid_state: GridAbstraction):
    """Get an adjacency graph of agent and objects in the scene. Objects are adjacent if there is a path that doesn't cross a door"""

    width, height = grid_state.grid.shape
    visited = np.zeros_like(grid_state.grid)

    objects = [(x, y, idx+1, obj_type == ObjectAbstraction.DOOR_IDX) for idx, (obj_type, x, y, _, _) in enumerate(grid_state.objects)]
    goal_index = len(objects) + 1
    positions = [grid_state.walker_pos, *[(x,y) for x, y, _, _ in objects]]

    regions_to_visit = deque([(grid_state.walker_pos, 0)]) # start from the agent position
    # indices_in_region = [0] # add agent to region
    # visited[grid_state.walker_pos] = 1 # visit agent square
    regions = []
    region_min_corners = []

    while len(regions_to_visit) > 0:
        # start at agent or door position
        region_seed, seed_object = regions_to_visit.popleft()
        min_corner = (np.inf, np.inf)

        if seed_object == 0:
            # agent
            indices_in_region = [0]
            new_regions = [region_seed]
            for x, y, _, is_door in objects:
                if is_door and (x, y) == region_seed:
                    # agent is on door; shift to side
                    new_regions = get_neighbors(region_seed, width, height)
            max_neighbors = 1
        else:
            indices_in_region = []
            new_regions = get_neighbors(region_seed, width, height)
            max_neighbors = 2
        # loop over adjacent squares
        for region_start in new_regions:
            # skip already visited
            if visited[region_start] == 1:
                continue
            if grid_state.grid[region_start] == GridAbstraction.GRID_WALL or grid_state.grid[region_start] == GridAbstraction.GRID_LAVA:
                    continue

            # visit all connected squares
            to_visit_in_region = deque([region_start])
            while len(to_visit_in_region) > 0:
                pos = to_visit_in_region.popleft()
                already_visited = visited[pos]
                if already_visited == 1:
                    continue
                visited[pos] = 1
                min_corner = min(pos, min_corner)
                if grid_state.grid[pos] == GridAbstraction.GRID_WALL or grid_state.grid[pos] == GridAbstraction.GRID_LAVA:
                    continue
                if grid_state.grid[pos] == GridAbstraction.GRID_GOAL:
                    indices_in_region.append(goal_index)
                    positions.append(pos)
                    goal_index += 1

                hit_door = False
                for x, y, obj_idx, is_door in objects:
                    # if object matches position, add to region indices
                    if (x, y) == pos:
                        indices_in_region.append(obj_idx)
                        if is_door:
                            hit_door = True
                            regions_to_visit.append((pos, obj_idx))
                            visited[pos] = 2 if already_visited == 0 else 1
                # don't go through door
                if not hit_door:
                    to_visit_in_region.extend(get_neighbors(pos, width, height))
            regions.append(indices_in_region)
            region_min_corners.append(min_corner)
            indices_in_region = [seed_object]
            max_neighbors -= 1
            if max_neighbors <= 0:
                break

    # sort rooms according to the smallest tile they contain, to give a consistent identifier
    regions = [x for _, x in sorted(zip(region_min_corners, regions), key=lambda pair: pair[0])]
    
    return regions, positions

def get_neighbors(pos, width, height):
    col, row = pos
    neighbors = [None, None, None, None]
    if col + 1 < width:
        # right
        neighbors[0] = (col+1, row)
    if row + 1 < height:
        # down
        neighbors[1] = (col, row+1)
    if col - 1 >= 0:
        # left
        neighbors[2] = (col-1, row)
    if row - 1 >= 0:
        # up
        neighbors[3] = (col, row-1)
    return [n for n in neighbors if n is not None]
