from __future__ import annotations

from typing import TYPE_CHECKING

from dm_control import composer

from cocogrid.dmc_gym import CocogridGym
from cocogrid.engine import Engine
from cocogrid.mujoco.cocogrid_arena import CocogridArena
from cocogrid.mujoco.cocogrid_task import CocogridTask

if TYPE_CHECKING:
    import gymnasium as gym

    from cocogrid.agent import Agent
    from cocogrid.state.observation import ObservationSpecification


class MujocoEngine(Engine):
    """Manage constructing MuJoCo physics environments."""

    @property
    def name(self) -> str:
        """Get the physics engine name."""
        return "MuJoCo"

    def build_gymnasium_env(self,
            minigrid: gym.Env,
            agent: Agent | type[Agent],
            get_task_function: callable,
            observation_spec: ObservationSpecification,
            render_mode: str = "rgb_array",
            render_width: int = 64,
            **env_kwargs: dict) -> gym.Env:
        """Build a MuJoCo gymnasium environment."""
        env_kwargs["get_task_function"] = get_task_function
        dmc_env = self.build_dmcontrol_env(minigrid, agent, **env_kwargs)

        return CocogridGym(
            dmc_env=dmc_env,
            observation_spec=observation_spec,
            rendering=None,
            render_width=render_width,
            render_mode=render_mode,
        )


    def build_dmcontrol_env(
            self, minigrid: gym.Env, agent: Agent | type[Agent], seed: int | None = None,
            timesteps: int = 500, **env_kwargs: dict) -> gym.Env:
        """Build a dm_control MuJoCo environment."""
        environment_kwargs = env_kwargs or {}
        task_kwargs = {}
        for key in ["observation_type", "random_rotation", "timesteps", "get_task_function"]:
            if key in environment_kwargs:
                task_kwargs[key] = environment_kwargs.pop(key)

        environment_kwargs["seed"] = seed

        arena = CocogridArena(minigrid, **environment_kwargs)

        if isinstance(agent, type):
            agent = agent(arena)

        physics_timestep = 0.005
        control_timestep = 0.03
        time_limit = control_timestep * timesteps - 0.00001  # subtract a tiny amount due to precision error

        task = CocogridTask(
            agent=agent,
            cocogrid_arena=arena,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            contact_termination=False,
            **task_kwargs,
        )

        return composer.Environment(
            task=task,
            time_limit=time_limit,
            random_state=seed,
            strip_singleton_obs_buffer_dim=True,
        )
