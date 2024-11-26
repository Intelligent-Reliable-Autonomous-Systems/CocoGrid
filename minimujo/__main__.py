import argparse

from minimujo import minimujo_suite

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', '-i', action='store_true', help='Spawns a dm_control interactive viewer. Requires GLFW.')
parser.add_argument('--gym', '-g', action='store_true', help='Runs HumanRendering with gymnasium.')
parser.add_argument('--minigrid', '-m', action='store_true', help='Run a minigrid environment')
parser.add_argument('--list', '-l', action='store_true', help='Print a list of environment ids.')
parser.add_argument('--env', '-e', type=str, default='Minimujo-Empty-5x5-v0', help='Specifies the Minimujo environment id.')
parser.add_argument('--detail', '-d', action='store_true', help='Gives detail about an environment.')
parser.add_argument('--framerate', '-f', action='store_true', help='Measures the framerate of the Minimujo environment.')
parser.add_argument('--obs-type', '-o', type=str, default='full', help="What type of observation should the environment emit? Options are 'top_camera', 'walker', 'pos'")
parser.add_argument('--img-obs-format', default='0-255', help="What format should image outputs be? Options are '0-255' (uint8) and '0-1' (float)")
parser.add_argument('--walker', '-w', type=str, default='ball', help="The type of the walker, from 'ball', 'ant', 'humanoid'")
parser.add_argument('--scale', '-s', type=float, default=1, help="The arena scale (minimum based on walker type)")
parser.add_argument('--timesteps', '-t', type=int, default=200, help="The maximum number of timesteps before truncating")
parser.add_argument('--random-spawn', action='store_true', help='The walker is randomly positioned on reset')
parser.add_argument('--random-rotate', action='store_true', help='The walker is randomly oriented on reset')
parser.add_argument('--track', action='store_true', help='when rendering gym, adds a trail behind the walker')
parser.add_argument('--seed', type=int, default=None, help='The random seed to be applied')
parser.add_argument('--episodes', type=int, default=1000, help="The number of episodes to run the gym env for")

parser.add_argument('--print-reward', action='store_true', help='Prints the reward and cumulative reward to the console')
parser.add_argument('--print-obs', action='store_true', help='Prints the observation to the console')
parser.add_argument('--test-steps', type=int, default=1000, help='For framerate test, number of steps per test')
parser.add_argument('--num-tests', type=int, default=5, help='For framerate test, the number of tests to run')
parser.add_argument('--sync-vec', type=int, default=0, help='If more than 0, how many environments to create in a SyncVectorEnv for framerate')
parser.add_argument('--async-vec', type=int, default=0, help='If more than 0, how many environments to create in an AsyncVectorEnv for framerate')
args = parser.parse_args()

long_dash = "-----------------------------------------"
print(long_dash)

def ensure_env():
    if args.env not in minimujo_suite.SUITE.keys():
        if f'Minimujo-{args.env}' in minimujo_suite.SUITE.keys():
            args.env = f'Minimujo-{args.env}'
            return
        print(f'Warning: {args.env} not a Minimujo environment')
        # raise Exception(f"Cannot spawn interactive session with invalid environment, {args.env}")

def get_gym_env(args):
    import minimujo
    return gym.make(
        args.env, 
        seed=args.seed, 
        walker_type=args.walker, 
        image_observation_format=args.img_obs_format, 
        observation_type=args.obs_type, 
        xy_scale=args.scale, 
        random_spawn=args.random_spawn, 
        random_rotation=args.random_rotate,
        timesteps=args.timesteps
    )

if args.interactive:
    import os
    import dm_control.suite as suite
    from dm_control import viewer
    import numpy as np

    ensure_env()

    env = suite.load('minimujo', args.env, task_kwargs={'walker_type': args.walker, 'random': args.seed}, environment_kwargs={'observation_type': args.obs_type, 'xy_scale': args.scale})

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
    
    env = get_gym_env(args)
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480

    # print('Environment with observables ', env.unwrapped._env._task.observables.keys())

    # from minimujo.utils.logging import LoggingWrapper, HeatmapLogger, get_minimujo_heatmap_loggers
    # import tensorboardX
    # summary_writer = tensorboardX.SummaryWriter()
    # env = LoggingWrapper(env, summary_writer, max_timesteps=args.timesteps)
    # for logger in get_minimujo_heatmap_loggers(env, gamma=0.99):
    #     env.subscribe_metric(logger)
    # from minimujo.state.goal_wrapper import GridPositionGoalWrapper
    # env = GridPositionGoalWrapper(env, dense=True, env_id=args.env)
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
        return np.array([grab, -up, right])

    obs, _ = env.reset()

    print(f'Env has observation space {env.unwrapped.observation_space} and action space {env.unwrapped.action_space}')
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    num_episodes = 0
    is_holding_reset = False
    is_eval = False
    while True:
        keys = key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print('Cumulative reward (to this point):', reward_sum)
            print('Manually terminated')
            break
        if keys[pygame.K_r] and not is_holding_reset:
            trunc = True
            is_holding_reset = True
        elif keys[pygame.K_e] and not is_holding_reset:
            is_eval = not is_eval
            trunc = True
            is_holding_reset = True
            print(f'Switched to eval={is_eval}')
        else:
            if not keys[pygame.K_r] and not keys[pygame.K_e]:
                is_holding_reset = False
            action = env.unwrapped.action_space.sample()
            manual_action = get_action()
            action[:3] = manual_action

            obs, rew, term, trunc, info = env.step(action)
            reward_sum += rew
            num_steps += 1

            if args.print_reward:
                print('reward:', rew)

            if args.print_obs:
                print('obs:', obs)
            
        if term or trunc:
            trunc_or_term = 'Truncated' if trunc else 'Terminated'
            print('Cumulative reward:', reward_sum)
            print(f'{trunc_or_term} after {num_steps} steps')
            num_episodes += 1
            if num_episodes >= args.episodes:
                break
            env.reset(options={'eval': is_eval})
            reward_sum = 0
            num_steps = 0
            term = trunc = False
            print(f"Current task: {env.unwrapped.task}")
    # print(env.unwrapped.range_mapping)

elif args.minigrid:
    import gymnasium as gym
    from minimujo.utils.minigrid import ManualControl

    # "left": Actions.left,
    # "right": Actions.right,
    # "up": Actions.forward,
    # "space": Actions.toggle,
    # "pageup": Actions.pickup,
    # "pagedown": Actions.drop,
    # "tab": Actions.pickup,
    # "left shift": Actions.drop,
    # "enter": Actions.done,
    env_id = args.env.replace('Minimujo', 'MiniGrid')
    env = gym.make(
        env_id,
        render_mode="human",
        screen_size=640,
    )

    manual_control = ManualControl(env)
    try:
        manual_control.start()
    except:
        print('Manual control terminated.')

elif args.list:
    print("\nSuite:")
    # for task_id in minimujo_suite.SUITE.keys():
    #     print(task_id)
    print(list(minimujo_suite.SUITE.keys()))

elif args.detail:
    import gymnasium as gym
    from minimujo.custom_minigrid import CUSTOM_ENVS

    ensure_env()

    minigrid_env_id = args.env.replace('Minimujo','MiniGrid')
    minigrid_env = gym.make(minigrid_env_id).unwrapped

    minigrid_env.reset(seed=args.seed)

    if type(minigrid_env) in CUSTOM_ENVS:
        print(f'{args.env} is a custom environment part of the Minimujo package\n')
    else:
        print(f'{args.env} corresponds to MiniGrid environment {minigrid_env_id}\n')

        print(f'Find details at https://minigrid.farama.org/environments/minigrid/{type(minigrid_env).__name__}\n')

    print('Here is a representation of the MiniGrid environment:')
    print(minigrid_env)

elif args.framerate:
    import gymnasium as gym
    import timeit

    N_STEPS = args.test_steps
    N_TESTS = args.num_tests
    ensure_env()
    
    if args.sync_vec > 0:
        from gymnasium.vector import SyncVectorEnv
        num_envs = args.sync_vec
        env = SyncVectorEnv([lambda: get_gym_env(args)] * args.sync_vec)
    elif args.async_vec > 0:
        from gymnasium.vector import AsyncVectorEnv
        num_envs = args.async_vec
        env = AsyncVectorEnv([lambda: get_gym_env(args)] * args.async_vec)
    else:
        num_envs = 1
        env = get_gym_env(args)

    def run_env():
        print(f'Testing gym env {args.env} with observation_type {args.obs_type} for {N_STEPS} steps')
        env.reset()

        for _ in range(0, N_STEPS, num_envs):
            action = env.action_space.sample()
            _, reward, term, trunc, _ = env.step(action)
            if reward is None:
                print('WARNING: reward returned was None')
            if isinstance(term, bool) and (term or trunc):
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
    print("    env = gymnasium.make(minimujo_id, **environment_kwargs)")

    print('\nBrowse MiniGrid environments at https://minigrid.farama.org/environments/minigrid/')
    print("Simply swap out the 'MiniGrid' for 'Minimujo' (e.g. MiniGrid-Empty-5x5-v0 -> Minimujo-Empty-5x5-v0)")
    print('Or select one from `python -m minimujo -l`')

    print("\nSee more options with `python -m minimujo --help`")

print(long_dash)