"""Register Cocogrid environments to gymnasium."""

import sys

import cocogrid.suite as suite
from cocogrid.agent import AgentRegistry, register_agents


def register_all() -> None:
    """Register all CocoGrid agents, environments, and tasks."""
    if not suite.IS_INITIALIZED:
        suite.register_environments_and_tasks()
    if not AgentRegistry.is_initialized() and "gymnasium" in sys.modules:
        register_agents()

register_all()
