"""Register Cocogrid environments to gymnasium."""

from cocogrid.agent import register_agents
from cocogrid.suite import register_environments_and_tasks

register_agents()
register_environments_and_tasks()
