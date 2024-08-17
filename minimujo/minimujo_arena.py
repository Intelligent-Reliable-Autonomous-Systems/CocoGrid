import itertools
import math

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import mazes
import labmaze
from labmaze import defaults as labdefaults
from minigrid.core.world_object import Door, Key, Ball, Box, Goal, Lava, Wall
import numpy as np

from minimujo.entities.ball_entity import BallEntity
from minimujo.entities.box_entity import BoxEntity
from minimujo.entities.contact_tile_entity import ContactTileEntity
from minimujo.entities.door_entity import DoorEntity
from minimujo.entities.key_entity import KeyEntity
from minimujo.grabber import Grabber
from minimujo.maze_arena import MazeArena
from minimujo.state.minimujo_state import MinimujoStateObserver
from minimujo.utils.minigrid import get_labmaze_from_minigrid

class MinimujoArena(MazeArena):
    def __init__(self, minigrid, xy_scale=1, z_height=2.0, cam_width=320, cam_height=240,
            random_spawn=False, spawn_padding=0.3, spawn_position=None, spawn_sampler=None, use_subgoal_rewards=False, dense_rewards=False, seed=None):
        """Initializes goal-directed minigrid task.
            Args:
            walker: The body to navigate the maze.
            maze_arena: The minigrid task environment.
            physics_timestep: timestep of simulation.
            control_timestep: timestep at which agent changes action.
        """

        self._minigrid = minigrid.unwrapped
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.random_spawn = random_spawn
        self.spawn_padding = spawn_padding
        self.spawn_position = spawn_position
        self.spawn_sampler = spawn_sampler
        self._minigrid_seed = seed

        labmaze = get_labmaze_from_minigrid(self._minigrid)
        
        super().__init__(
            maze=labmaze,
            xy_scale=xy_scale,
            z_height=z_height,
            name='minimujo'
        )
        self._already_initialized = True # don't regenerate maze the first time
        
        # gives the walker a magnet to move objects
        self._grabber = Grabber()
        self.attach(self._grabber)

        self.create_entity_mjcf()
        self.current_state = None

        self._walker_position = np.array([0,0,0])
        self._terminated = False
        self._extrinsic_reward = 0
        self._dense_rewards = dense_rewards

    def _build_observables(self):
        return MinimujoObservables(self, self.cam_width, self.cam_height)
    
    def create_entity_mjcf(self):
        if hasattr(self, '_mini_entity_map') and isinstance(self._mini_entity_map, dict):
            for entity_list in self._mini_entity_map.values():
                for entry in entity_list:
                    entity = entry['entity']
                    if isinstance(entity, composer.Entity):
                        entity.detach()

        self._mini_entity_map = {
            Ball: [],
            Box: [],
            Door: [],
            Key: [],
            Goal: [],
            Lava: []
        }
        width = self._minigrid.grid.width
        for idx, miniObj in enumerate(self._minigrid.grid.grid):
            if miniObj is not None:
                key = type(miniObj)
                if key not in self._mini_entity_map:
                    self._mini_entity_map[key] = []
                row = idx // width
                col = idx % width
                color = miniObj.color
                world_pos = self.grid_to_world_positions([(row,col)])[0]
                if key is Door:
                    entity = DoorEntity(color=color, is_locked=miniObj.is_locked, xy_scale=self.xy_scale)
                    self.attach(entity)
                elif key is Key:
                    entity = KeyEntity(self._grabber, color=color)
                    entity.create_root_joints(self.attach(entity))
                elif key is Ball:
                    entity = BallEntity(self._grabber, color=color)
                    entity.create_root_joints(self.attach(entity))
                elif key is Box:
                    entity = BoxEntity(self._grabber, color=color)
                    entity.create_root_joints(self.attach(entity))
                elif key is Goal:
                    entity = ContactTileEntity(color='goal_green', xy_scale=self.xy_scale)
                    self.attach(entity)
                elif key is Lava:
                    entity = ContactTileEntity(color='orange', xy_scale=self.xy_scale)
                    self.attach(entity)
                else:
                    entity = None
                    continue
                self._mini_entity_map[key].append({
                    'entity': entity, 
                    'world_pos': world_pos, 
                    'mini_pos': (col, row), 
                    'mini_obj': miniObj
                })

        self.setDoorDirections(self._maze.entity_layer)

    def setDoorDirections(self, charMatrix):
        # find an empty space surrounding the door
        # assume the that opposite side of that empty space is also empty
        for door in self._mini_entity_map[Door]:
            dir = 0
            c, r = door['mini_pos']
            if r - 1 >= 0 and charMatrix[r-1][c] == '*':
                # up
                dir = 0
            elif r + 1 < len(charMatrix[0]) and charMatrix[r-1][c] == '*':
                # down
                dir = 2
            if c - 1 >= 0 and charMatrix[r][c-1] == '*':
                # left
                dir = 1
            elif c + 1 < len(charMatrix) and charMatrix[r][c+1] == '*':
                # right
                dir = 3
            door['dir'] = dir

    def get_state_observer(self):
        filtered_entities = [entities for key, entities in self._mini_entity_map.items() 
            if key in [Key, Ball, Box, Door]]
        objects = [entity['entity'] for entity in itertools.chain(*filtered_entities)]
        return MinimujoStateObserver(self._minigrid, self.xy_scale, objects, self._walker)

    def initialize_arena_mjcf(self, random_state=None):
        if not self._already_initialized:
            self._minigrid.reset(seed=self._minigrid_seed)
        self._maze = get_labmaze_from_minigrid(self._minigrid)
        self.regenerate()
        
        if not self._already_initialized:
            self.create_entity_mjcf()
            self.register_walker(self._walker, self._get_walker_pos)
        self._already_initialized = False

        self.state_observer = self.get_state_observer()
        self.current_state = None
    
    def initialize_arena(self, physics, random_state):
        if self.spawn_position is not None:
            pos = np.array(self.spawn_position)
            pos.resize((3,))
            self._spawn_positions = [pos]
        elif self.spawn_sampler is not None:
            pos = np.array(self.spawn_sampler())
            pos.resize((3,))
            self._spawn_positions = [pos]
        elif self.random_spawn:
            self._spawn_positions = (self.get_random_spawn_position(),)

        for door in self._mini_entity_map[Door]:
            door['entity'].set_position(physics, door['world_pos'], door['dir'])
            door['entity'].reset()
            for obj in self._mini_entity_map[Key]:
                door['entity'].register_key(obj['entity'])

        for obj_type in self._mini_entity_map.keys():
            if obj_type is Wall:
                continue
            for obj in self._mini_entity_map[obj_type]:
                if 'entity' not in obj or obj['entity'] is None:
                    print('no entity', obj)
                    continue
                obj['entity'].set_pose(physics, position=obj['world_pos'])   
        
        self._walker_position = self._spawn_positions[0]

        # self._minigrid_manager.reset()
        # self._minigrid_manager.sync_minigrid(self)
        self._terminated = False
        self._extrinsic_reward = 0

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

    def register_walker(self, walker, get_walker_pos):
        self._walker = walker
        self._get_walker_pos = get_walker_pos
        self._grabber.register_walker(self._walker)
        for lava in self._mini_entity_map[Lava]:
            lava['entity'].register_walker(walker)
        for goal in self._mini_entity_map[Goal]:
            goal['entity'].register_walker(walker)

    def before_step(self, physics, random_state):
        if self.current_state is None:
            self.current_state = self.state_observer.get_state(physics)
    
    def update_state(self, physics):
        self.previous_state = self.current_state
        self.current_state = self.state_observer.get_state(physics)

    @property
    def walker_position(self):
        return self.current_state.get_walker_position()
    
    @property
    def walker_grid_position(self):
        row, col = self.world_to_minigrid_position(self.walker_position)
        return col, row
    
    @property
    def walker_grid_continuous_position(self):
        return self.world_to_minigrid_continuous_position(self.walker_position)
    
    @property
    def maze_width(self):
        return self._minigrid.grid.width * self.xy_scale
    
    @property
    def maze_height(self):
        return self._minigrid.grid.height * self.xy_scale
    
    def world_to_minigrid_position(self, position):
        row, col = self.world_to_grid_positions([position])[0]
        # col, row are integers in center of tile, so need to round to nearest integer. e.g -.6->1, 1.4->1
        return int(row + 1/2), int(col + 1/2)
    
    def world_to_minigrid_continuous_position(self, position):
        return self.world_to_grid_positions([position])[0]
    
    def minigrid_to_world_position(self, row, col):
        # returns center position of tile
        return self.grid_to_world_positions([(row, col)])[0]
    
    def maze_bounds(self):
        x_low = self._xy_scale * -self._x_offset
        y_low = self._xy_scale * -self._y_offset
        x_high = self._xy_scale * (-self._x_offset + self._maze.width)
        y_high = self._xy_scale * (-self._y_offset + self._maze.height)
        return x_low, y_low, x_high, y_high
    
    def get_random_spawn_position(self):
        MAX_TRIES = 100
        x_low, y_low, x_high, y_high = self.maze_bounds()
        for _ in range(MAX_TRIES):
            x = np.random.uniform(x_low, x_high)
            y = np.random.uniform(y_low, y_high)
            test_coords = [
                (x-self.spawn_padding, y-self.spawn_padding),
                (x-self.spawn_padding, y+self.spawn_padding),
                (x+self.spawn_padding, y-self.spawn_padding),
                (x+self.spawn_padding, y+self.spawn_padding),
            ]
            is_valid = True
            for test_x, test_y in test_coords:
                row, col = self.world_to_minigrid_position((test_x, test_y, 0))
                try:
                    if self._minigrid.grid.get(col, row) is not None:
                        # is overlapping something
                        is_valid = False
                        break
                except:
                    # was out of bounds
                    is_valid = False
                    break
            if is_valid:
                row, col = self.world_to_minigrid_position((x, y, 0))
                self._minigrid.agent_pos = (col, row)
                # if not self._minigrid_manager.has_solution():
                #     is_valid = False
            if is_valid:
                return np.array([x, y, 0])
        
        # backup spawn position
        col, row = self._minigrid.agent_pos
        return self.grid_to_world_positions(((row, col),))[0]
    
class MinimujoObservables(composer.Observables):

  def __init__(self, entity, width=240, height=320):
    self.width = width
    self.height = height
    super().__init__(entity)

  @composer.observable
  def top_camera(self):
    return observable.MJCFCamera(self._entity.top_camera, width=self.width, height=self.height)