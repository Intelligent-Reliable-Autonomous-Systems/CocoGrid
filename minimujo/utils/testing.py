import argparse

def add_minimujo_arguments(parser: argparse.ArgumentParser, include_seed=False, **defaults):
    parser.add_argument('--env', '-e', type=str, default=defaults.get('env', 'Minimujo-Empty-5x5-v0'), help='Specifies the Minimujo environment id.')
    parser.add_argument('--walker', '-w', type=str, default=defaults.get('walker', 'ball'), help="The type of the walker, from 'ball', 'ant', 'humanoid'")
    parser.add_argument('--scale', '-s', type=float, default=defaults.get('scale', 1), help="The arena scale (minimum based on walker type)")
    parser.add_argument('--timesteps', '-t', type=int, default=defaults.get('timesteps', 200), help="The maximum number of timesteps before truncating")
    parser.add_argument('--random-spawn', action='store_true', help='The walker is randomly positioned on reset')
    parser.add_argument('--random-rotate', action='store_true', help='The walker is randomly oriented on reset')
    if include_seed:
        parser.add_argument('--seed', type=int, default=defaults.get('seed', None), help='The random seed to be applied')

def args_to_gym_env(args, **env_kwargs):
    import minimujo
    import gymnasium
    return gymnasium.make(
        args.env,
        walker_type=args.walker,
        seed=args.seed if hasattr(args, 'seed') else None,
        xy_scale=args.scale,
        random_spawn=args.random_spawn,
        random_rotation=args.random_rotate,
        timesteps=args.timesteps,
        **env_kwargs
    )

def get_pygame_action():
    import pygame
    import numpy as np
    keys = pygame.key.get_pressed()
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

def run_gymnasium_env(env, print_reward=False, print_obs=False):
    import pygame

    obs, _ = env.reset()

    print(f'Env has observation space {env.unwrapped.observation_space} and action space {env.unwrapped.action_space}')
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    num_episodes = 0
    is_holding_reset = False
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print('Cumulative reward (to this point):', reward_sum)
            print('Manually terminated')
            break
        if keys[pygame.K_r] and not is_holding_reset:
            trunc = True
            is_holding_reset = True
        else:
            if not keys[pygame.K_r]:
                is_holding_reset = False
            action = env.unwrapped.action_space.sample()
            manual_action = get_pygame_action()
            action[:3] = manual_action


            obs, rew, term, trunc, info = env.step(action)
            reward_sum += rew
            num_steps += 1

            if print_reward:
                print('reward:', rew)

            if print_obs:
                print('obs:', obs)
            
        if term or trunc:
            trunc_or_term = 'Truncated' if trunc else 'Terminated'
            print('Cumulative reward:', reward_sum)
            print(f'{trunc_or_term} after {num_steps} steps')
            num_episodes += 1
            env.reset()
            reward_sum = 0
            num_steps = 0
            term = trunc = False
            # print(f"Current task: {env.unwrapped.task}")