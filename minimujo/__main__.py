import argparse

from minimujo import minimujo_suite

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', '-i', action='store_true', help='Spawns a dm_control interactive viewer. Requires GLFW.')
parser.add_argument('--gym', '-g', action='store_true', help='Runs HumanRendering with gymnasium.')
parser.add_argument('--list', '-l', action='store_true', help='Print a list of environment ids.')
parser.add_argument('--env', '-e', type=str, default='Minimujo-Empty-5x5-v0', help='Specifies the Minimujo environment id.')
parser.add_argument('--detail', '-d', action='store_true', help='Gives detail about an environment.')
parser.add_argument('--framerate', '-f', action='store_true', help='Measures the framerate of the Minimujo environment.')
parser.add_argument('--obs-type', '-o', default='top_camera', help="What type of output should the environment have? Options are 'top_camera', 'walker', 'pos'")
args = parser.parse_args()

long_dash = "-----------------------------------------"
print(long_dash)

def ensure_env():
    if args.env not in minimujo_suite.SUITE.keys():
        raise Exception(f"Cannot spawn interactive session with invalid environment, {args.env}")

if args.interactive:
    import os
    import dm_control.suite as suite
    from dm_control import viewer
    import numpy as np

    ensure_env()

    env = suite.load('minimujo', args.env, environment_kwargs={'observation_type': args.obs_type})

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

elif args.gym:
    import gymnasium as gym
    from gymnasium.wrappers.human_rendering import HumanRendering
    import numpy as np
    from pygame import key
    import pygame

    ensure_env()
    
    env = gym.make(args.env, env_params={'observation_type': args.obs_type})
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480
    env = HumanRendering(env)

    print('Controls: Move with WASD, grab with Space')

    def get_action():
        keys = key.get_pressed()
        up = 0
        right = 0
        grab = 0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            right += 1
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            right -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            up -= 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            up += 1
        if keys[pygame.K_n] or keys[pygame.K_SPACE]:
            grab = 1
        if keys[pygame.K_ESCAPE]:
            return None
        return np.array([grab, -up, right])

    env.reset()

    num_steps = 0
    while True:
        action = get_action()
        if action is None:
            break
        obs, rew, term, trunc, info = env.step(action)
        num_steps += 1
        if term:
            print(f'Terminated after {num_steps} steps')
            break
        if trunc:
            print(f'Truncated after {num_steps} steps')
            break

elif args.list:
    print("\nSuite:")
    # for task_id in minimujo_suite.SUITE.keys():
    #     print(task_id)
    print(list(minimujo_suite.SUITE.keys()))

elif args.detail:
    import gymnasium as gym

    ensure_env()

    minigrid_env_id = args.env.replace('Minimujo','MiniGrid')
    minigrid_env = gym.make(minigrid_env_id).unwrapped

    minigrid_env.reset()

    print(f'{args.env} corresponds to MiniGrid environment {minigrid_env_id}\n')

    print(f'Find details at https://minigrid.farama.org/environments/minigrid/{type(minigrid_env).__name__}\n')

    print('Here is a representation of the MiniGrid environment:')
    print(minigrid_env)

elif args.framerate:
    import gymnasium as gym
    import timeit

    N_STEPS = 1000
    N_TESTS = 10
    ensure_env()
    
    env = gym.make(args.env, env_params={'observation_type': args.obs_type})

    def run_env():
        print(f'Testing gym env {args.env} with observation_type {args.obs_type} for {N_STEPS} steps')
        env.reset()

        for _ in range(N_STEPS):
            action = env.action_space.sample()
            _, reward, term, trunc, _ = env.step(action)
            if reward is None:
                print('WARNING: reward returned was None')
            if term or trunc:
                env.reset()

    total_time = timeit.timeit("run_env()", globals=locals(), number=N_TESTS)
    fps = 1 / (total_time / N_TESTS / N_STEPS)
    print(f'Average FPS = {fps}')

else:
    print("Minimujo: Continuous navigation environment\n")
    print("Run dm_env environments like:")
    print("    import dm_control.suite as suite")
    print("    env = suite.load('minimujo', minimujo_id, environment_kwargs={...})")
    print("\nOr with gymnasium as:")
    print("    env = gym.make(minimujo_id, env_params=env_params)")

    print('\nBrowse MiniGrid environments at https://minigrid.farama.org/environments/minigrid/')
    print("Simply swap out the 'MiniGrid' for 'Minimujo' (e.g. MiniGrid-Empty-5x5-v0 -> Minimujo-Empty-5x5-v0)")
    print('Or select one from `python -m minimujo -l`')

    print("\nSee more options with `python -m minimujo --help`")

print(long_dash)