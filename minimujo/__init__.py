import functools
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from dm_control.suite import _DOMAINS
from gymnasium.envs.registration import register
from minimujo import minimujo_suite

minimujo_domain = 'minimujo'
if not minimujo_domain in _DOMAINS:
    _DOMAINS[minimujo_domain] = minimujo_suite

    for task_id in minimujo_suite.SUITE.keys():
        register(
            id=task_id,
            entry_point=functools.partial(minimujo_suite.get_gym_env_from_suite, minimujo_domain, task_id)
        )