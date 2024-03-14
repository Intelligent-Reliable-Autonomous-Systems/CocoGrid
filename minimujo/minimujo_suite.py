import functools

from dm_control import composer
from dm_control.locomotion.walkers import ant, cmu_humanoid, jumping_ball
from dm_control.utils import containers
from minimujo.dmc_gym import DMCGym
import gymnasium
from gymnasium.envs.registration import registry

from minimujo.custom_minigrid import register_custom_minigrid
from minimujo.minimujo_arena import MinimujoArena
from minimujo.minimujo_task import MinimujoTask

def get_minimujo_env(minigrid_id, walker_type='rolling_ball', time_limit=20, random=None, environment_kwargs=None):
    highEnv = gymnasium.make(minigrid_id)
    highEnv.reset()

    environment_kwargs = environment_kwargs or {}

    if walker_type == 'rolling_ball' or walker_type == 'ball':
        walker = jumping_ball.RollingBallWithHead(initializer=tuple())
    elif walker_type == 'cmu_humanoid' or walker_type == 'human' or walker_type == 'humanoid':
        walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
            observable_options={'egocentric_camera': dict(enabled=True)})
    elif walker_type == 'ant':
        walker = ant.Ant()
        environment_kwargs['xy_scale'] = max(2, environment_kwargs.get('xy_scale', 2))
    else:
        raise Exception(f'walker_type {walker_type} not supported')

    task_keys = ['observation_type']
    task_kwargs = {}
    for key in task_keys:
        if key in environment_kwargs:
            task_kwargs[key] = environment_kwargs.pop(key)

    arena = MinimujoArena(highEnv.unwrapped, **environment_kwargs)

    task = MinimujoTask(
        walker=walker,
        minimujo_arena=arena,
        physics_timestep=0.005,
        control_timestep=0.03,
        contact_termination=False,
        **task_kwargs
    )

    env = composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=random,
        strip_singleton_obs_buffer_dim=True,
    )
    return env

def get_gym_env_from_suite(domain, task, time_limit=20, random=None, env_params=None):
    return DMCGym(domain=domain, task=task, task_kwargs=dict(time_limit=time_limit, random=random), environment_kwargs=env_params, rendering=None)

SUITE = containers.TaggedTasks()

register_custom_minigrid()
minigrid_env_ids = [env_spec.id for env_spec in registry.values() if env_spec.id.startswith("MiniGrid")]

for minigrid_id in minigrid_env_ids:
    minimujo_id = minigrid_id.replace("MiniGrid", "Minimujo")
    SUITE._tasks[minimujo_id] = functools.partial(get_minimujo_env, minigrid_id)

def get_minimujo_env_ids():
    return SUITE._tasks.keys()