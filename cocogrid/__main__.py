"""Cocogrid CLI tool.

This module provides a command-line interface for interacting with Cocogrid environments.
It supports both gymnasium and dm_control environments, as well as MiniGrid environments.

Available commands:
    interactive (i): Spawn a dm_control interactive viewer with 3D visualization
    gym (g): Run a gymnasium environment with manual control
    minigrid (m): Run a MiniGrid environment
    list (l): List available Cocogrid environments
    detail (d): Get detailed information about an environment
    framerate (f): Measure environment performance

Examples:
    # List all environments
    python -m cocogrid list

    # Run an interactive environment
    python -m cocogrid interactive --env Empty-5x5

    # Run a gym environment
    python -m cocogrid gym --env Empty-5x5 --episodes 10

    # Get details about an environment
    python -m cocogrid detail --env Empty-5x5

    # Filter environments
    python -m cocogrid list "empty,door"  # Show environments with "empty" or "door"
    python -m cocogrid list "!door"       # Exclude environments with "door"
"""

import argparse

import gymnasium as gym

# from cocogrid import cocogrid_suite
from cocogrid.suite import REGISTERED_GYM_IDS
from cocogrid.utils.testing import add_cocogrid_arguments


def ensure_env(args: argparse.Namespace) -> None:
    """Check for variants of the environment name and mutate args.env."""
    if args.env not in REGISTERED_GYM_IDS:
        if f"Cocogrid-{args.env}" in REGISTERED_GYM_IDS:
            args.env = f"Cocogrid-{args.env}"
        elif f"Cocogrid-{args.env}-v0" in REGISTERED_GYM_IDS:
            args.env = f"Cocogrid-{args.env}-v0"
        elif f"{args.env}-v0" in REGISTERED_GYM_IDS:
            args.env = f"{args.env}-v0"
        else:
            print(f"Warning: {args.env} not a Cocogrid environment")


def get_gym_env(args: argparse.Namespace) -> gym.Env:
    """Construct a gym environment from CLI arguments."""
    return gym.make(
        args.env,
        seed=args.seed,
        walker_type=args.walker,
        observation_type=args.obs,
        xy_scale=args.scale,
        random_spawn=args.random_spawn,
        random_rotation=args.random_rotate,
        timesteps=args.timesteps,
    )


def run_interactive_environment(args: argparse.Namespace) -> None:
    """Run an interactive MuJoCo environment with a 3D view."""
    import os

    import numpy as np
    from dm_control import suite, viewer

    ensure_env(args)

    env = suite.load(
        "cocogrid",
        args.env,
        task_kwargs={
            "walker_type": args.walker,
            "seed": args.seed,
        },
        environment_kwargs={
            "observation_type": args.obs,
            "xy_scale": args.scale,
        },
    )

    os.environ["MUJOCO_GL"] = "glfw"
    import glfw

    def policy(time: float) -> np.ndarray:
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


def run_gym_environment(args: argparse.Namespace) -> None:
    """Run a gymnasium environment with manual control."""
    import numpy as np
    import pygame
    from gymnasium.wrappers.human_rendering import HumanRendering
    from pygame import key

    ensure_env(args)

    env = get_gym_env(args)
    env.unwrapped.render_width = 480
    env.unwrapped.render_height = 480

    env = HumanRendering(env)

    print("Controls: Move with WASD, grab with Space")

    def get_action() -> np.ndarray:
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

    print(
        f"Env has observation space {env.unwrapped.observation_space} and action space {{env.unwrapped.action_space}}",
    )
    print(f"Current task: {env.unwrapped.task}")

    num_steps = 0
    reward_sum = 0
    num_episodes = 0
    is_holding_reset = False
    is_eval = False
    while True:
        keys = key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print("Cumulative reward (to this point):", reward_sum)
            print("Manually terminated")
            break
        if keys[pygame.K_r] and not is_holding_reset:
            trunc = True
            is_holding_reset = True
        elif keys[pygame.K_e] and not is_holding_reset:
            is_eval = not is_eval
            trunc = True
            is_holding_reset = True
            print(f"Switched to eval={is_eval}")
        else:
            if not keys[pygame.K_r] and not keys[pygame.K_e]:
                is_holding_reset = False
            action = env.unwrapped.action_space.sample()
            manual_action = get_action()
            action[:3] = manual_action

            obs, rew, term, trunc, _ = env.step(action)
            reward_sum += rew
            num_steps += 1

            if args.print_reward:
                print("reward:", rew)

            if args.print_obs:
                print("obs:", obs)

        if term or trunc:
            trunc_or_term = "Truncated" if trunc else "Terminated"
            print("Cumulative reward:", reward_sum)
            print(f"{trunc_or_term} after {num_steps} steps")
            num_episodes += 1
            if num_episodes >= args.episodes:
                break
            env.reset(options={"eval": is_eval})
            reward_sum = 0
            num_steps = 0
            term = trunc = False
            print(f"Current task: {env.unwrapped.task}")
    # print(env.unwrapped.range_mapping)


def run_minigrid(args: argparse.Namespace) -> None:
    """Run a MiniGrid environment."""
    from cocogrid.utils.minigrid import ManualControl

    #  The key actions of MiniGrid are:
    #  "left": Actions.left
    #  "right": Actions.right
    #  "up": Actions.forward
    #  "space": Actions.toggle
    #  "pageup": Actions.pickup
    #  "pagedown": Actions.drop
    #  "tab": Actions.pickup
    #  "left shift": Actions.drop,
    #  "enter": Actions.done
    env_id = args.env.replace("Cocogrid", "MiniGrid")
    env = gym.make(
        env_id,
        render_mode="human",
        screen_size=640,
        highlight=False,
    )

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()


def list_environments(args: argparse.Namespace) -> None:
    """List available Cocogrid evironments."""
    print("\nSuite:")
    # envs = list(cocogrid_suite.SUITE.keys())
    envs = REGISTERED_GYM_IDS

    def sensitivity(s: str) -> str:
        if args.sensitive:
            return s
        return s.lower()

    # Parse filters
    filters = args.filters.split(",") if args.filters else []
    include_filters = {sensitivity(f) for f in filters if not f.startswith("!")}
    exclude_filters = {sensitivity(f[1:]) for f in filters if f.startswith("!")}

    # Apply inclusion filtering (if any include filters exist, require at least one match)
    if include_filters:
        envs = [env for env in envs if any(f in sensitivity(env) for f in include_filters)]

    # Apply exclusion filtering (remove matches for exclude filters)
    envs = [env for env in envs if not any(f in sensitivity(env) for f in exclude_filters)]

    if args.vertical:
        print("\n".join(envs))
    else:
        print(envs)


def get_details(args: argparse.Namespace) -> None:
    """Get details about an environment."""
    from cocogrid.custom_minigrid import CUSTOM_ENVS

    ensure_env(args)

    minigrid_env_id = args.env.replace("Cocogrid", "MiniGrid")
    minigrid_env = gym.make(minigrid_env_id).unwrapped

    minigrid_env.reset(seed=args.seed)

    if type(minigrid_env) in CUSTOM_ENVS:
        print(f"{args.env} is a custom environment part of the Cocogrid package\n")
    else:
        print(f"{args.env} corresponds to MiniGrid environment {minigrid_env_id}\n")

        print(
            f"Find details at https://minigrid.farama.org/environments/minigrid/{type(minigrid_env).__name__}\n",
        )

    print("Here is a representation of the MiniGrid environment:")
    print(minigrid_env)


def test_framerate(args: argparse.Namespace) -> None:
    """Run a number of episodes to measure steps per second."""
    import timeit

    n_steps = args.test_steps
    n_tests = args.num_tests
    ensure_env(args)

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

    def run_env() -> None:
        """Run the environment for N_STEPS."""
        print(
            f"Testing gym env {args.env} with observation_type {args.obs} for {{N_STEPS}} steps",
        )
        env.reset()

        for _ in range(0, n_steps, num_envs):
            action = env.action_space.sample()
            _, reward, term, trunc, _ = env.step(action)
            if reward is None:
                print("WARNING: reward returned was None")
            if isinstance(term, bool) and (term or trunc):
                env.reset()

    total_time = timeit.timeit("run_env()", globals=locals(), number=n_tests)
    fps = 1 / (total_time / n_tests / n_steps)
    print(f"Average FPS = {fps}")


def print_helpful() -> None:
    """Print a helpful message how to use Cocogrid."""
    print("Cocogrid: Continuous navigation environment\n")
    print("Run gymnasium environments like:")
    print("    env = gymnasium.make(cocogrid_id, **environment_kwargs)")
    print("Or dm_control environments as:")
    print("    import dm_control.suite as suite")
    print("    env = suite.load('cocogrid', cocogrid_id, environment_kwargs={...})")

    print(
        "\nBrowse MiniGrid environments at https://minigrid.farama.org/environments/minigrid/",
    )
    print(
        "Simply swap out the 'MiniGrid' for 'Cocogrid' (e.g. MiniGrid-Empty-5x5-v0 -> Cocogrid-Empty-5x5-v0)",
    )
    print("Or select one from `python -m cocogrid -l`")

    print("\nSee more options with `python -m cocogrid --help`")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Cocogrid: Continuous navigation environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive subcommand
    interactive_parser = subparsers.add_parser(
        "interactive", aliases=["i"], help="Spawns a dm_control interactive viewer. Requires GLFW."
    )
    add_cocogrid_arguments(interactive_parser, include_seed=True)

    # Gym subcommand
    gym_parser = subparsers.add_parser("gym", aliases=["g"], help="Runs HumanRendering with gymnasium.")
    add_cocogrid_arguments(gym_parser, include_seed=True)
    gym_parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="The number of episodes to run the gym env for",
    )
    gym_parser.add_argument(
        "--print-reward",
        action="store_true",
        help="Prints the reward and cumulative reward to the console",
    )
    gym_parser.add_argument(
        "--print-obs",
        action="store_true",
        help="Prints the observation to the console",
    )

    # Minigrid subcommand
    minigrid_parser = subparsers.add_parser("minigrid", aliases=["m"], help="Run a minigrid environment")
    add_cocogrid_arguments(minigrid_parser, include_seed=True)

    # List subcommand
    list_parser = subparsers.add_parser("list", aliases=["l"], help="Print a list of environment ids.")
    list_parser.add_argument(
        "filters", nargs="?", type=str, help="Comma-separated filter keywords (supports negations with '!')"
    )
    list_parser.add_argument("--sensitive", "-s", action="store_true", help="Make filter case-sensitive.")
    list_parser.add_argument(
        "--vertical", "--vert", "-v", action="store_true", help="Display environments line by line"
    )

    # Detail subcommand
    detail_parser = subparsers.add_parser("detail", aliases=["d"], help="Gives detail about an environment.")
    add_cocogrid_arguments(detail_parser, include_seed=True)

    # Framerate subcommand
    framerate_parser = subparsers.add_parser(
        "framerate", aliases=["f"], help="Measures the framerate of the Cocogrid environment."
    )
    add_cocogrid_arguments(framerate_parser, include_seed=True)
    framerate_parser.add_argument(
        "--test-steps",
        type=int,
        default=1000,
        help="For framerate test, number of steps per test",
    )
    framerate_parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="For framerate test, the number of tests to run",
    )
    framerate_parser.add_argument(
        "--sync-vec",
        type=int,
        default=0,
        help="Number of environments to create in SyncVectorEnv for framerate test",
    )
    framerate_parser.add_argument(
        "--async-vec",
        type=int,
        default=0,
        help="Number of environments to create in AsyncVectorEnv for framerate test",
    )

    return parser.parse_args()


def main() -> None:
    """Run the main CLI tool."""
    args = parse_args()

    long_dash = "-----------------------------------------"
    print(long_dash)

    if args.command in ["interactive", "i"]:
        run_interactive_environment(args)
    elif args.command in ["gym", "g"]:
        run_gym_environment(args)
    elif args.command in ["minigrid", "m"]:
        run_minigrid(args)
    elif args.command in ["list", "l"]:
        list_environments(args)
    elif args.command in ["detail", "d"]:
        get_details(args)
    elif args.command in ["framerate", "f"]:
        test_framerate(args)
    else:
        print_helpful()

    print(long_dash)


if __name__ == "__main__":
    main()
