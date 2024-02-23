import dm_control.suite as suite
import numpy as np
import minimujo
import gymnasium

# highEnv = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0")
# highEnv = gymnasium.make("MiniGrid-KeyCorridorS6R3-v0")
# highEnv = gymnasium.make("MiniGrid-Playground-v0")
# gym_env = gymnasium.make('Minimujo-LavaCrossingS9N2-v0')
# gym_env.reset()
# gym_env.step(np.array([0, 0, 0]))

env = suite.load('minimujo', 'Minimujo-Playground-v0')

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