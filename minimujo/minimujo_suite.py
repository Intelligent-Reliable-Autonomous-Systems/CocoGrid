import functools

from dm_control import composer
from dm_control.locomotion.walkers import ant, cmu_humanoid, jumping_ball
from dm_control.locomotion.walkers.base import Walker
from dm_control.utils import containers
from minimujo.dmc_gym import MinimujoGym
import gymnasium
from gymnasium.envs.registration import registry

from minimujo.custom_minigrid import register_custom_minigrid, default_tasks
from minimujo.minimujo_arena import MinimujoArena
from minimujo.minimujo_task import MinimujoTask
from minimujo.walkers.square import Square
from minimujo.walkers.rolling_ball import RollingBallWithHead
from minimujo.walkers.ant import Ant
from minimujo.state.tasks import get_grid_goal_task

def get_minimujo_env(minigrid_id, walker_type='square', timesteps=500, seed=None, environment_kwargs=None):
    highEnv = gymnasium.make(minigrid_id)
    highEnv.reset(seed=seed)

    environment_kwargs = environment_kwargs or {}
    task_kwargs = {}
    task_keys = ['observation_type', 'random_rotation', 'task_function', 'get_task_function']
    for key in task_keys:
        if key in environment_kwargs:
            task_kwargs[key] = environment_kwargs.pop(key)
    
    if 'reward_type' in environment_kwargs:
        print('Deprecation Warning: reward_type is unused. specify the reward in the task function')
        environment_kwargs.pop('reward_type')

    if walker_type == 'rolling_ball' or walker_type == 'ball':
        walker = RollingBallWithHead(initializer=tuple())
    elif walker_type == 'cmu_humanoid' or walker_type == 'human' or walker_type == 'humanoid':
        walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
            observable_options={'egocentric_camera': dict(enabled=True)})
    elif walker_type == 'ant':
        walker = ant.Ant()
        # walker = Ant()
        environment_kwargs['xy_scale'] = max(2, environment_kwargs.get('xy_scale', 2))
        environment_kwargs['spawn_padding'] = 0.8
    elif walker_type == 'square':
        walker = Square()
        task_kwargs['random_rotation'] = False
    elif isinstance(walker_type, Walker):
        walker = walker_type
    else:
        raise Exception(f'walker_type {walker_type} not supported')

    environment_kwargs['seed'] = seed

    arena = MinimujoArena(highEnv.unwrapped, **environment_kwargs)

    PHYSICS_TIMESTEP=0.005
    CONTROL_TIMESTEP=0.03
    time_limit = CONTROL_TIMESTEP * timesteps - 0.00001 # subtrack a tiny amount due to precision error

    if 'task_function' in task_kwargs:
        # the task function is the same every episode
        task_function = task_kwargs.pop('task_function')
        task_kwargs['get_task_function'] = lambda minigrid: (task_function, '')
    if not 'get_task_function' in task_kwargs:
        task_kwargs['get_task_function'] = default_task_registry.get(type(highEnv.unwrapped), get_grid_goal_task)
    task = MinimujoTask(
        walker=walker,
        minimujo_arena=arena,
        physics_timestep=PHYSICS_TIMESTEP,
        control_timestep=CONTROL_TIMESTEP,
        contact_termination=False,
        **task_kwargs
    )

    env = composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
    )
    return env

def get_gym_env_from_suite(domain, task, walker_type='ball', observation_type='no-arena', image_observation_format='0-255', timesteps=200, seed=None, track_position=False, render_mode='rgb_array', render_width=64, **env_kwargs):
    if 'box2d' in walker_type.lower():
        return get_box2d_gym_env(task, walker_type, observation_type=observation_type, image_observation_format=image_observation_format, timesteps=timesteps, seed=seed, render_width=render_width, **env_kwargs)
    return MinimujoGym(
        domain=domain, 
        task=task, 
        task_kwargs=dict(walker_type=walker_type, timesteps=timesteps, seed=seed), 
        environment_kwargs=env_kwargs, 
        observation_type=observation_type,
        image_observation_format=image_observation_format,
        rendering=None, 
        render_width=render_width,
        render_mode=render_mode,
        track_position=track_position)

def get_box2d_gym_env(arena_id, walker_type, observation_type=None, task_function = None, get_task_function = None, **env_kwargs):
    from minimujo.box2d.gym import Box2DEnv
    minigrid_id = arena_id.replace('Minimujo', 'MiniGrid')
    minigrid_env = gymnasium.make(minigrid_id).unwrapped

    if task_function is not None:
        # the task function is the same every episode
        get_task_function = lambda minigrid: (task_function, '')
    elif get_task_function is None:
        get_task_function = default_task_registry.get(type(minigrid_env), get_grid_goal_task)
    return Box2DEnv(minigrid_env, walker_type, get_task_function, observation_type=observation_type, **env_kwargs)

SUITE = containers.TaggedTasks()

register_custom_minigrid()
minigrid_env_ids = [env_spec.id for env_spec in registry.values() if env_spec.id.startswith("MiniGrid")]

default_task_registry = {**default_tasks}

for minigrid_id in minigrid_env_ids:
    minimujo_id = minigrid_id.replace("MiniGrid", "Minimujo")
    SUITE._tasks[minimujo_id] = functools.partial(get_minimujo_env, minigrid_id)

def get_minimujo_env_ids():
    return SUITE._tasks.keys()