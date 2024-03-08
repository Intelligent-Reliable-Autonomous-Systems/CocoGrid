import argparse

from minimujo import minimujo_suite

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', '-i', action='store_true', help='Spawns an interactive viewer. Requires GLFW.')
parser.add_argument('--env', '-e', type=str, default='Minimujo-Empty-5x5-v0', help='The Minimujo environment id')
args = parser.parse_args()

long_dash = "-----------------------------------------"
print(long_dash)
print("Minimujo: Continuous navigation environment\n")
print("Run dmc_env environments like:")
print("    import dm_control.suite as suite")
print("    env = suite.load('minimujo', gym_id, environment_kwargs={...})")
print("\nOr with gymnasium as:")
print("    env = gym.make(gym_id, env_params=env_params)")
# print("\nSuite:")
# for task_id in minimujo_suite.SUITE.keys():
#     print(task_id)
# print(list(minimujo_suite.SUITE.keys()))

print("\nSee more options with `python -m minimujo --help`")
print(long_dash)

if args.interactive:
    import os
    import dm_control.suite as suite
    from dm_control import viewer
    import numpy as np

    if args.env not in minimujo_suite.SUITE.keys():
        raise Exception(f"Cannot spawn interactive session with invalid environment, {args.env}")

    env = suite.load('minimujo', args.env)

    # os.environ['PYOPENGL_PLATFORM'] = 'glfw'
    os.environ['MUJOCO_GL'] = 'glfw'
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

    viewer.launch(env, policy=policy)
