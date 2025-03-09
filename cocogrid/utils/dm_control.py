import functools
from dm_control import suite
from gymnasium.envs.registration import register

from cocogrid.dmc_gym import DMCGym

def get_gym_env_from_suite(domain, task, task_kwargs=None, env_kwargs=None, image_observation_format='0-255', render_mode='rgb_array', render_width=64):
    return DMCGym(
        domain=domain, 
        task=task, 
        task_kwargs=task_kwargs, 
        environment_kwargs=env_kwargs, 
        image_observation_format=image_observation_format,
        rendering=None, 
        render_width=render_width,
        render_mode=render_mode)

def register_domain_in_gym(domain):
    assert domain in suite._DOMAINS, f"Cannot register domain {domain} to gym. It could not be found."

    domain_suite = suite._DOMAINS[domain].SUITE
    for task in domain_suite.keys():
        task_id = f'{domain}-{task}'
        register(task_id, functools.partial(get_gym_env_from_suite, domain, task))

def register_manipulation_in_gym():
    from dm_control import manipulation

    def get_manipulation_gym_env(task, image_observation_format='0-255', render_mode='rgb_array', render_width=64):
        dmc_env = manipulation.load(task)
        return DMCGym(
            domain='manipulation', 
            task=task,
            dmc_env=dmc_env,
            image_observation_format=image_observation_format,
            rendering=None, 
            render_width=render_width,
            render_mode=render_mode)
    
    for task in manipulation.ALL:
        task_hyphen = task.replace('_', '-')
        task_id = f'manipulation-{task_hyphen}'
        register(task_id, functools.partial(get_manipulation_gym_env, task))