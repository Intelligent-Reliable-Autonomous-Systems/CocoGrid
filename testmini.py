import minigrid
import gymnasium
import labmaze
from labmaze import defaults as labdefaults
from dm_control import mjcf
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.composer.variation import distributions
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.props import target_sphere
from dm_control import composer
from dm_control import mjcf
import PIL.Image
import numpy as np

# Set the wall and floor textures to match DMLab and set the skybox.
skybox_texture = labmaze_textures.SkyBox(style='sky_03')
wall_textures = labmaze_textures.WallTextures(style='style_01')
floor_textures = labmaze_textures.FloorTextures(style='style_01')


# highEnv = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0")
highEnv = gymnasium.make("MiniGrid-KeyCorridorS6R3-v0")
highEnv.reset()

# print(highEnv.unwrapped.grid.grid)
gridStr = str(highEnv.unwrapped)

key = {
  'W': '*',
  '^': labdefaults.SPAWN_TOKEN,
  '>': labdefaults.SPAWN_TOKEN,
  '<': labdefaults.SPAWN_TOKEN,
  'V': labdefaults.SPAWN_TOKEN,
  'A': labdefaults.OBJECT_TOKEN
}
wallEntities = [
    ''.join([ key[c] if c in key else c for c in row[::2]])
    for row in gridStr.split('\n')
]
# print(labdefaults.SPAWN_TOKEN)
labmazeStr = '\n'.join(wallEntities) + '\n'
maze = labmaze.FixedMazeWithRandomGoals(labmazeStr)
print(maze.entity_layer)
# highEnv.pprint_grid()
# print([method_name for method_name in dir(highEnv.unwrapped)
#                   if callable(getattr(highEnv.unwrapped, method_name))])

# highEnv.unwrapped.pprint_grid()

# print(dir(type(highEnv.unwrapped).__bases__[0]))
walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
    observable_options={'egocentric_camera': dict(enabled=True)})

arena = mazes.MazeWithTargets(
    maze=maze,
    xy_scale=1,
    skybox_texture=skybox_texture,
    wall_textures=wall_textures,
    floor_textures=floor_textures)
arena._find_spawn_and_target_positions()
print(arena.find_token_grid_positions([labdefaults.SPAWN_TOKEN]))
print(arena.target_positions)

# arena = corridor_arenas.WallsCorridor(
#     wall_gap=3.,
#     wall_width=distributions.Uniform(2., 3.),
#     wall_height=distributions.Uniform(2.5, 3.5),
#     corridor_width=4.,
#     corridor_length=30.,
# )
# arena = mazes.RandomMazeWithTargets(
#     x_cells=11,
#     y_cells=11,
#     xy_scale=3,
#     max_rooms=4,
#     room_min_size=4,
#     room_max_size=5,
#     spawns_per_room=1,
#     targets_per_room=3,
#     skybox_texture=skybox_texture,
#     wall_textures=wall_textures,
#     floor_textures=floor_textures)

# task = corridor_tasks.RunThroughCorridor(
#     walker=walker,
#     arena=arena,
#     walker_spawn_position=(0.5, 0, 0),
#     target_velocity=3.0,
#     physics_timestep=0.005,
#     control_timestep=0.03,
# )
task = random_goal_maze.RepeatSingleGoalMaze(
  walker=walker,
  maze_arena=arena,
  physics_timestep=0.005,
  control_timestep=0.03
)

env = composer.Environment(
    task=task,
    time_limit=10,
    random_state=np.random.RandomState(42),
    strip_singleton_obs_buffer_dim=True,
)
env.reset()
testObj = target_sphere.TargetSphere(radius=0.9)
task.root_entity.attach(testObj)
mjcf.get_attachment_frame(testObj.mjcf_model).pos = arena.target_positions[0]
spec = env.action_spec()
random_state = np.random.RandomState(42)
action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
env.step(action)
pixels = []
for camera_id in range(3):
  pixels.append(env.physics.render(camera_id=camera_id, width=240))
img = PIL.Image.fromarray(np.hstack(pixels))
img.save('test.png')