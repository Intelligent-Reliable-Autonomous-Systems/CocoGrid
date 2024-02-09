import math
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.props import target_sphere
from dm_control import mjcf
from dm_control import composer
import labmaze
from labmaze import defaults as labdefaults
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes

from minigrid.core.world_object import Door, Key, Ball


from abstractcontrol.door_entity import DoorEntity
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

        gridStr = str(self._minigrid)

        key = {
        'W': '*',
        '^': labdefaults.SPAWN_TOKEN,
        '>': labdefaults.SPAWN_TOKEN,
        '<': labdefaults.SPAWN_TOKEN,
        'V': labdefaults.SPAWN_TOKEN,
        'A': labdefaults.OBJECT_TOKEN
        }
        charMatrix = [
            [ key[c] if c in key else c for c in row[::2]]
            for row in gridStr.split('\n')
        ]

        labmazeStr = '\n'.join([''.join(row) for row in charMatrix]) + '\n'
        self._labmaze = labmaze.FixedMazeWithRandomGoals(labmazeStr)
        print(self._labmaze.entity_layer)
        
        super().__init__(
            maze=self._labmaze,
            xy_scale=xy_scale,
            z_height=z_height,
            skybox_texture=skybox_texture,
            wall_textures=wall_textures,
            floor_textures=floor_textures)

        self._mini_entity_map = {}
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
                    entity = KeyEntity(color=color)
                    entity.create_root_joints(self.attach(entity))
                elif key is Ball:
                    entity = target_sphere.TargetSphere(radius=5)
                    self.attach(entity)
                else:
                    entity = None
                self._mini_entity_map[key].append({
                    'entity': entity, 
                    'world_pos': world_pos, 
                    'mini_pos': (row, col), 
                    'mini_obj': miniObj
                })

        self.getDoorDirections(charMatrix)
        
        # d_positions = [(r, c) for r, row in enumerate(charMatrix) for c, char in enumerate(row) if char == 'D']
        # for door in self._mini_entity_map[Door]:
        #     self.attach(door['entity'])

        # for key in self._mini_entity_map[Key]:
        #     self.attach(key['entity'])

        # self._doors = self.generateDoors(charMatrix)
        # for door in self._doors:
        #     self.attach(door[0])

        # self._keys = self.generateKeys(charMatrix)
        # for key in self._keys:
        #     self.attach(key[0])

        # for door in self._mini_entity_map[Door]:
        #     door[0] = DoorEntity()

        # for key in self._mini_entity_map[Key]:
        #     key[0] = KeyEntity()
            # mjcf.get_attachment_frame(door[0]).pos = door[1]

        # print(d_positions)
        self._target2 = target_sphere.TargetSphere(radius=3)
        self.attach(self._target2)

    def getDoorDirections(self, charMatrix):
        # d_positions = [(r, c) for r, row in enumerate(charMatrix) for c, char in enumerate(row) if char == 'D']

        # doors = []
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
            # world_pos = self.grid_to_world_positions([(r,c)])[0]

            # doors.append((DoorEntity(), world_pos, dir))
            door['dir'] = dir
    
    def generateKeys(self, charMatrix):
        k_positions = [(r, c) for r, row in enumerate(charMatrix) for c, char in enumerate(row) if char == 'K']

        keys = []
        for r, c in k_positions:
            world_pos = self.grid_to_world_positions([(r,c)])[0]

            keys.append((KeyEntity(), world_pos))

        return keys
    
    def initialize_arena(self, physics, random_state):
        door_quats = [
            [1, 0, 0, 0],
            [math.sin(math.pi/4), 0, 0, math.sin(math.pi/4)],
            [0, 0, 0, 1],
            [math.sin(math.pi/4), 0, 0, -math.sin(math.pi/4)]
        ]
        # for door in self._doors:
        #     door[0].set_pose(physics, position=door[1], quaternion=door_quats[door[2]])

        # for key in self._keys:
        #     key[0].set_pose(physics, position=key[1])

        for door in self._mini_entity_map[Door]:
            door['entity'].set_pose(physics, position=door['world_pos'], quaternion=door_quats[door['dir']])

        for key in self._mini_entity_map[Key]:
            key['entity'].set_pose(physics, position=key['world_pos'])

        for ball in self._mini_entity_map[Ball]:
            ball['entity'].set_pose(physics, position=ball['world_pos'])
            # mjcf.get_attachment_frame(
            #     ball['entity']).pos = ball['world_pos']
            


    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        self._target_position = self.target_positions[
            random_state.randint(0, len(self.target_positions))]
        print(self._target_position, 'targ pos')
        mjcf.get_attachment_frame(
            self._target2.mjcf_model).pos = self._target_position
        # for ball in self._mini_entity_map[Ball]:
        #     # ball['entity'].set_pose(physics, position=ball['world_pos'])
        #     mjcf.get_attachment_frame(
        #         ball['entity']).pos = ball['world_pos']
    
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