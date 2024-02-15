import math
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.props import target_sphere
from dm_control import mjcf
from dm_control import composer
import labmaze
from labmaze import defaults as labdefaults
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes

from minigrid.core.world_object import Door, Key, Ball, Box, Goal, Lava, Wall
from abstractcontrol.ball_entity import BallEntity
from abstractcontrol.box_entity import BoxEntity
from abstractcontrol.contact_tile_entity import ContactTileEntity


from abstractcontrol.door_entity import DoorEntity
from abstractcontrol.grabber import Grabber
from abstractcontrol.key_entity import KeyEntity

DEFAULT_PHYSICS_TIMESTEP = 0.001
DEFAULT_CONTROL_TIMESTEP = 0.025
DEFAULT_SKYBOX_TEXTURE = labmaze_textures.SkyBox(style='sky_03')
DEFAULT_WALL_TEXTURE = labmaze_textures.WallTextures(style='style_01')
DEFAULT_FLOOR_TEXTURE = labmaze_textures.FloorTextures(style='style_01')

            # physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
            # control_timestep=DEFAULT_CONTROL_TIMESTEP
        # self.set_timesteps(
        #     physics_timestep=physics_timestep, control_timestep=control_timestep)

class MinimujoArena(mazes.MazeWithTargets):
    def __init__(self, minigrid, xy_scale=1, z_height=2.0, name='minimujo',
            skybox_texture=DEFAULT_SKYBOX_TEXTURE, 
            wall_textures=DEFAULT_WALL_TEXTURE, 
            floor_textures=DEFAULT_FLOOR_TEXTURE):
        """Initializes goal-directed minigrid task.
            Args:
            walker: The body to navigate the maze.
            maze_arena: The minigrid task environment.
            physics_timestep: timestep of simulation.
            control_timestep: timestep at which agent changes action.
        """

        # self._xy_scale = xy_scale
        # self._z_height = z_height

        self._minigrid = minigrid.unwrapped

        # gridStr = str(self._minigrid)

        # key = {
        #     'W': '*',
        #     '^': labdefaults.SPAWN_TOKEN,
        #     '>': labdefaults.SPAWN_TOKEN,
        #     '<': labdefaults.SPAWN_TOKEN,
        #     'V': labdefaults.SPAWN_TOKEN,
        #     'A': labdefaults.OBJECT_TOKEN,
        #     'P': 'L'
        # }
        # charMatrix = [
        #     [ key[c] if c in key else c for c in row[::2]]
        #     for row in gridStr.split('\n')
        # ]

        # labmazeStr = '\n'.join([''.join(row) for row in charMatrix]) + '\n'

        maze_width = self._minigrid.grid.width
        maze_height = self._minigrid.grid.height
        walls = ['*' if type(s) is Wall else ' ' for s in self._minigrid.grid.grid]
        if self._minigrid.agent_pos is not None:
            c, r = self._minigrid.agent_pos
            walls[r * maze_width + c] = labdefaults.SPAWN_TOKEN
        labmaze_matrix = [walls[i:i+maze_width] for i in range(0, len(walls), maze_width)]
        labmaze_str = '\n'.join([''.join(row) for row in labmaze_matrix]) + '\n'

        self._labmaze = labmaze.FixedMazeWithRandomGoals(labmaze_str)
        print(self._labmaze.entity_layer)
        
        super().__init__(
            maze=self._labmaze,
            xy_scale=xy_scale,
            z_height=z_height,
            skybox_texture=skybox_texture,
            wall_textures=wall_textures,
            floor_textures=floor_textures)
        
        # actuator = self._mjcf_root.actuator
        # dummy = self._mjcf_root.worldbody.add('body', name='dummy')
        # dummy.add('joint', name='grab', type='slide', axis=[1, 0, 0])
        # print(actuator)
        # self.grab_control = actuator.add('general', name="grab", joint="grab", ctrlrange=[0, 1], ctrllimited=True, gear="30")
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
                    entity = DoorEntity(color=color, is_locked=miniObj.is_locked)
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
                    'mini_pos': (row, col), 
                    'mini_obj': miniObj
                })
        print(self._mini_entity_map.keys())

        self.getDoorDirections(labmaze_matrix)

    def getDoorDirections(self, charMatrix):
        for door in self._mini_entity_map[Door]:
            dir = 0
            r, c = door['mini_pos']
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

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

    def register_walker(self, walker):
        self._walker = walker
        self._grabber.register_walker(self._walker)
        for lava in self._mini_entity_map[Lava]:
            lava['entity'].register_walker(walker)
        for goal in self._mini_entity_map[Goal]:
            goal['entity'].register_walker(walker)


    
    # def _build_observables(self):
    #     return self._arena._build_observables()

    # @property
    # def top_camera(self):
    #     return self._arena._top_camera

    # @property
    # def xy_scale(self):
    #     return self._xy_scale

    # @property
    # def z_height(self):
    #     return self._z_height

    # @property
    # def maze(self):
    #     return self._arena