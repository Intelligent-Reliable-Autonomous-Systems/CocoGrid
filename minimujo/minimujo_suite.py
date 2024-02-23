import functools

from dm_control import composer
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.walkers import jumping_ball
from dm_control.utils import containers
from dmc2gymnasium import DMCGym
import gymnasium
from gymnasium.envs.registration import registry

from minimujo.minimujo_arena import MinimujoArena
from minimujo.minimujo_task import MinimujoTask

def get_minimujo_env(minigrid_id, walker_type, time_limit=20, random=None, environment_kwargs=None):
    highEnv = gymnasium.make(minigrid_id)
    highEnv.reset()

    if walker_type == 'rolling_ball':
        walker = jumping_ball.RollingBallWithHead()
    elif walker_type == 'cmu_humanoid':
        walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
            observable_options={'egocentric_camera': dict(enabled=True)})
    else:
        raise Exception(f'walker_type {walker_type} not supported')
    
    environment_kwargs = environment_kwargs or {}

    arena = MinimujoArena(highEnv.unwrapped, **environment_kwargs)

    task = MinimujoTask(
        walker=walker,
        minimujo_arena=arena,
        physics_timestep=0.005,
        control_timestep=0.03,
        contact_termination=False,
    )

    env = composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=random,
        strip_singleton_obs_buffer_dim=True,
    )
    return env

def get_gym_env_from_suite(domain, task, time_limit=20, random=None, environment_kwargs=None):
    return DMCGym(domain=domain, task=task, task_kwargs=dict(time_limit=time_limit, random=random), environment_kwargs=environment_kwargs)

SUITE = containers.TaggedTasks()

minigrid_env_ids = [env_spec.id for env_spec in registry.values() if env_spec.id.startswith("MiniGrid")]

for minigrid_id in minigrid_env_ids:
    minimujo_id = minigrid_id.replace("MiniGrid", "Minimujo")
    SUITE._tasks[minimujo_id] = functools.partial(get_minimujo_env, minigrid_id, 'rolling_ball')