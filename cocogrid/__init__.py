"""Register Cocogrid environments to gymnasium."""

import functools
import os
import sys
import types
from typing import Callable

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import cocogrid.suite as suite
from cocogrid.agent import AgentRegistry, register_agents


def register_all() -> None:
    """Register all CocoGrid agents, environments, and tasks."""
    if not suite.IS_INITIALIZED:
        suite.register_environments_and_tasks()
    if not AgentRegistry.is_initialized() and "gymnasium" in sys.modules:
        register_agents()

def gymnasium_entrypoint() -> None:
    """Register cocogrid when called from gymnasium.

    The entrypoint connected to 'gymnasium.envs' which will be called by 'import gymnasium'.
    Minigrid is a dependency, so must be loaded first. We must do some disgusting importlib hackery.
    minigrid.register_minigrid_envs is replaced with a duplicate and the original is mutated to a no-op.
        This function should only be called once anyway, so not the worst.
    """
    from importlib.metadata import entry_points

    from gymnasium.envs import registration

    is_minigrid_loaded = False
    for env_id in registration.registry:
        if env_id.lower().startswith("minigrid"):
            is_minigrid_loaded = True

    if not is_minigrid_loaded:
        # Find minigrid entry point
        eps = entry_points(group="gymnasium.envs")
        try:
            minigrid_entry = next(entry for entry in eps if entry.value.startswith("minigrid"))
        except StopIteration:
            raise Exception("Minigrid is a CocoGrid dependency, but no entrypoint was found.")
        # Temporarily replace the gymnasium namespace.
        # current_namespace is the only side effect of the namespace context in gymnasium.envs.registration.
        prev_namespace = registration.current_namespace
        registration.current_namespace = None if minigrid_entry.name == "__root__" else minigrid_entry.name
        minigrid_entry.load()()
        registration.current_namespace = prev_namespace
        # Save a copy of minigrid.register_minigrid_envs
        import minigrid
        duplicate_func = copy_func(minigrid_entry.load())
        minigrid.register_minigrid_envs = duplicate_func
        # Mutate the original function call because Entrypoint is immutable.
        def noop() -> None:
            pass
        minigrid_entry.load().__code__ = noop.__code__

    # Finally, register CocoGrid
    register_all()

def copy_func(f: Callable) -> Callable:
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)."""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
