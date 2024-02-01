import PIL.Image
import labmaze
from dm_control.locomotion.arenas import MazeWithTargets
from dm_control.locomotion.arenas import labmaze_textures
import numpy as np

with open('abstractcontrol/maze.txt', 'r') as maze:
    mazeStr = maze.read()

labMaze = labmaze.FixedMazeWithRandomGoals(
    entity_layer=mazeStr
)

# Set the wall and floor textures to match DMLab and set the skybox.
skybox_texture = labmaze_textures.SkyBox(style='sky_03')
wall_textures = labmaze_textures.WallTextures(style='style_01')
floor_textures = labmaze_textures.FloorTextures(style='style_01')

maze_arena = MazeWithTargets(
    maze=labMaze,
    xy_scale=16,
    z_height=3,
    skybox_texture=skybox_texture,
    wall_textures=wall_textures,
    floor_textures=floor_textures,
    name='maze_test'
)

# Test the arena

from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control import composer
from utils import display_video

walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
    observable_options={'egocentric_camera': dict(enabled=True)})

mazeTask = random_goal_maze.NullGoalMaze(
    walker=walker,
    maze_arena=maze_arena,
    randomize_spawn_position=True,
    randomize_spawn_rotation=True
)

env = composer.Environment(
    task=mazeTask,
    time_limit=10,
    strip_singleton_obs_buffer_dim=True,
)
# env.reset()
# pixels = []
# for camera_id in range(3):
#   pixels.append(env.physics.render(camera_id=camera_id, width=240))
# image = PIL.Image.fromarray(np.hstack(pixels))
# image.show()

# action_spec = env.action_spec()

# def sample_random_action():
#   return env.random_state.uniform(
#       low=action_spec.minimum,
#       high=action_spec.maximum,
#   ).astype(action_spec.dtype, copy=False)

# Step the environment through a full episode using random actions and record
# the camera observations.
# frames = []
# timestep = env.reset()
# frames.append(timestep.observation['walker/egocentric_camera'])
# while not timestep.last():
#   timestep = env.step(sample_random_action())
#   frames.append(timestep.observation['walker/egocentric_camera'])
# all_frames = np.stack(frames, axis=0)
# display_video(all_frames, 30)

from dm_control import viewer

viewer.launch(env)