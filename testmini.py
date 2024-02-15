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
from dm_control.locomotion.walkers import jumping_ball
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.props import target_sphere
from dm_control import composer
from dm_control import mjcf
import PIL.Image
import numpy as np
from abstractcontrol.maze_task_test import MazeTaskTest

from abstractcontrol.minimujo_arena import MinimujoArena
from abstractcontrol.minimujo_task import MinimujoTask

# Set the wall and floor textures to match DMLab and set the skybox.
skybox_texture = labmaze_textures.SkyBox(style='sky_03')
wall_textures = labmaze_textures.WallTextures(style='style_01')
floor_textures = labmaze_textures.FloorTextures(style='style_01')


# highEnv = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0")
# highEnv = gymnasium.make("MiniGrid-KeyCorridorS6R3-v0")
highEnv = gymnasium.make("MiniGrid-Playground-v0")
highEnv.reset()

# print(dir(type(highEnv.unwrapped).__bases__[0]))
# walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
#     observable_options={'egocentric_camera': dict(enabled=True)})
walker = jumping_ball.RollingBallWithHead()

arena = MinimujoArena(highEnv.unwrapped, xy_scale=1)

task = MinimujoTask(
  walker=walker,
  minimujo_arena=arena,
  physics_timestep=0.005,
  control_timestep=0.03,
  contact_termination=False,
)

env = composer.Environment(
    task=task,
    time_limit=100,
    random_state=np.random.RandomState(42),
    strip_singleton_obs_buffer_dim=True,
)
env.reset()
# testObj = target_sphere.TargetSphere(radius=0.9)
# task.root_entity.attach(testObj)
# mjcf.get_attachment_frame(testObj.mjcf_model).pos = arena.target_positions[0]
# spec = env.action_spec()
# random_state = np.random.RandomState(42)
# action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
# env.step(action)
# pixels = []
# for camera_id in range(3):
#   pixels.append(env.physics.render(camera_id=camera_id, width=240))
# img = PIL.Image.fromarray(np.hstack(pixels))
# img.save('test.png')

print(env.action_spec())

import glfw

def policy(time):
    ctx = glfw.get_current_context()
    up = 0
    right = 0
    grab = 0
    if glfw.get_key(ctx, glfw.KEY_RIGHT):
        right += 1
    if glfw.get_key(ctx, glfw.KEY_LEFT):
        right -= 1
    if glfw.get_key(ctx, glfw.KEY_UP):
        up -= 1
    if glfw.get_key(ctx, glfw.KEY_DOWN):
        up += 1
    if glfw.get_key(ctx, glfw.KEY_N):
        grab = 1
    return np.array([grab, up, right])

from dm_control import viewer

viewer.launch(env, policy=policy)
# viewer.launch(env)