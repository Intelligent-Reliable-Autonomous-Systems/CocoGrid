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
from .minigrid.minigrid_manager import MinigridManager

class MinimujoArena(mazes.MazeWithTargets):
    def __init__(self, minigrid, xy_scale=1, z_height=2.0, cam_width=320, cam_height=240, name='minimujo',
            skybox_texture=None, 
            wall_textures=None, 
            floor_textures=None):
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

        maze_width = self._minigrid.grid.width
        walls = ['*' if type(s) is Wall else ' ' for s in self._minigrid.grid.grid]
        if self._minigrid.agent_pos is not None:
            c, r = self._minigrid.agent_pos
            walls[r * maze_width + c] = labdefaults.SPAWN_TOKEN
        labmaze_matrix = [walls[i:i+maze_width] for i in range(0, len(walls), maze_width)]
        labmaze_str = '\n'.join([''.join(row) for row in labmaze_matrix]) + '\n'

        self._labmaze = labmaze.FixedMazeWithRandomGoals(labmaze_str)
        
        super().__init__(
            maze=self._labmaze,
            xy_scale=xy_scale,
            z_height=z_height,
            skybox_texture=skybox_texture,
            wall_textures=wall_textures,
            floor_textures=floor_textures,
            name=name
        )
        
        self._grabber = Grabber()
        self.attach(self._grabber)

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
                    entity = DoorEntity(color=color, is_locked=miniObj.is_locked, xy_scale=xy_scale)
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
                    entity = ContactTileEntity(color='goal_green', xy_scale=xy_scale)
                    self.attach(entity)
                elif key is Lava:
                    entity = ContactTileEntity(color='orange', xy_scale=xy_scale)
                    self.attach(entity)
                else:
                    entity = None
                self._mini_entity_map[key].append({
                    'entity': entity, 
                    'world_pos': world_pos, 
                    'mini_pos': (col, row), 
                    'mini_obj': miniObj
                })

        self.setDoorDirections(labmaze_matrix)

        self._walker_position = np.array([0,0,0])
        self._minigrid_manager = MinigridManager(self._minigrid, self._mini_entity_map)
        self._terminated = False
        self._extrinsic_reward = 0

    def _build_observables(self):
        return MinimujoObservables(self, self.cam_width, self.cam_height)

    def setDoorDirections(self, charMatrix):
        for door in self._mini_entity_map[Door]:
            dir = 0
            c, r = door['mini_pos']
            if r - 1 >= 0 and charMatrix[r-1][c] == '*':
                dir = 0
            elif r + 1 < len(charMatrix[0]) and charMatrix[r-1][c] == '*':
                dir = 2
            if c - 1 >= 0 and charMatrix[r][c-1] == '*':
                dir = 1
            elif c + 1 < len(charMatrix) and charMatrix[r][c+1] == '*':
                dir = 3
            door['dir'] = dir
    
    def initialize_arena(self, physics, random_state):
        door_quats = [
            [1, 0, 0, 0],
            [math.sin(math.pi/4), 0, 0, math.sin(math.pi/4)],
            [0, 0, 0, 1],
            [math.sin(math.pi/4), 0, 0, -math.sin(math.pi/4)]
        ]

        for door in self._mini_entity_map[Door]:
            door['entity'].set_pose(physics, position=door['world_pos'], quaternion=door_quats[door['dir']])
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

        self._terminated = False
        self._extrinsic_reward = 0    

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

    def register_walker(self, walker):
        self._walker = walker
        self._grabber.register_walker(self._walker)
        for lava in self._mini_entity_map[Lava]:
            lava['entity'].register_walker(walker)
        for goal in self._mini_entity_map[Goal]:
            goal['entity'].register_walker(walker)

    def before_step(self, physics, random_state):
        if self._walker:
            self._walker_position = physics.bind(self._walker.root_body).xpos
        if not self._terminated:
            reward, terminated = self._minigrid_manager.sync_minigrid(self)
            self._terminated = terminated
            self._extrinsic_reward += reward

    @property
    def walker_position(self):
        return self._walker_position
    
    @property
    def walker_grid_position(self):
        return self.world_to_minigrid_position(self.walker_position)
    
    def world_to_minigrid_position(self, position):
        col, row = self.world_to_grid_positions([position])[0]
        # col, row are integers in center of tile, so need to round to nearest integer. e.g -.6->1, 1.4->1
        return int(row + 1/2), int(col + 1/2)
    
    def minigrid_to_world_position(self, row, col):
        position = self.grid_to_world_positions([(row, col)])[0]
        # returns center position of tile
        return position
    
class MinimujoObservables(composer.Observables):

  def __init__(self, entity, width=240, height=320):
    self.width = width
    self.height = height
    super().__init__(entity)

  @composer.observable
  def top_camera(self):
    return observable.MJCFCamera(self._entity.top_camera, width=self.width, height=self.height)