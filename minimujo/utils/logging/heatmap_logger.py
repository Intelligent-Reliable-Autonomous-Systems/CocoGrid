

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import warnings

import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tensorboardX import SummaryWriter

from minimujo.dmc_gym import DMCGym
from minimujo.minimujo_arena import MinimujoArena
from minimujo.utils.logging import LoggingMetric, capped_cubic_logging_schedule
from minimujo.utils.visualize.weighted_kde import WeightedKDEHeatmap

class HeatmapLogger(LoggingMetric):

    def __init__(self, label: str, step_transform: Callable, end_transform: Callable = None, logging_schedule: Callable = capped_cubic_logging_schedule, 
            should_log_density: bool = False, extent: Tuple[float] = (0, 1, 0, 1), decay: float = 1, axes_label: Tuple[str] = ('X', 'Y', 'Value'), color_map: str = 'plasma', 
            is_log_scale: bool = False, value_min: Optional[float] = None, value_max: Optional[float] = None, origin_corner: Literal['lower', 'upper'] = 'lower') -> None:
        self.transform = step_transform
        self.end_transform = end_transform or dummy_end_transform
        self.logging_schedule = logging_schedule

        xy_range = (extent[0], extent[2], extent[1]-extent[0], extent[3]-extent[2])
        self.heatmap = WeightedKDEHeatmap(xy_range=xy_range, decay=decay)

        # Figure properties
        self.axes_label = axes_label
        self.color_map = color_map
        self.origin_corner = origin_corner
        self.extent = extent
        self.label = label
        self.is_log_scale = is_log_scale
        self.norm = LogNorm() if is_log_scale else None
        self.should_log_density = should_log_density
        self.value_min = value_min
        self.value_max = value_max

        self.current_episode = 0

    def register(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int, global_step_callback: Callable = None):
        super().register(env, summary_writer, max_timesteps, global_step_callback)
        self.buffer = np.zeros((max_timesteps, 3))

    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        self.current_episode = episode

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        self.end_transform(batch=self.buffer[:timesteps], env=self.env, timestep=timesteps)

        filtered_buffer = self.buffer[:timesteps]
        filtered_buffer = filtered_buffer[np.all(np.isfinite(filtered_buffer), axis=1)]
        self.heatmap.add_batch(filtered_buffer[:,:2], filtered_buffer[:,2])

        if self.logging_schedule(episode):
            try:
                figure, _ = plt.subplots()
                logmap = self.heatmap.densitymap_normalized if self.should_log_density else self.heatmap.heatmap
                plt.imshow(logmap, origin=self.origin_corner, cmap=self.color_map, extent=self.extent, norm=self.norm, vmin=self.value_min, vmax=self.value_max)
                plt.colorbar(label=self.axes_label[2])
                plt.title(f"{self.label} (Episode {episode})")
                plt.xlabel(self.axes_label[0])
                plt.ylabel(self.axes_label[1])
                if self.summary_writer is not None:
                    self.summary_writer.add_figure(self.label, figure, self.global_step_callback(), True)
            except Exception as e:
                warnings.warn(f"Exception occurred in heatmap {self.label}: {str(e)}", UserWarning)

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        if timestep >= self.max_timesteps:
            return
        self.buffer[timestep] = np.asarray(self.transform(
            obs=obs, rew=rew, term=term, trunc=trunc, info=info, env=self.env, timestep=timestep
        ))

def get_minimujo_extent(env: gym.Env, without_border: bool = True):
    """Gets the extent of the heatmap figure, within the arena bounds"""
    env = env.unwrapped
    border_off = int(without_border)
    if isinstance(env, DMCGym):
        arena: MinimujoArena = env.arena
        extent = (border_off, arena.maze_width - border_off, -(arena.maze_height - border_off), -border_off)
    else:
        # box2d
        extent = (border_off, env.arena_width - border_off, -(env.arena_height - border_off), -border_off)
    return extent

def walker_xyz_transform(env, **kwargs):
    """Transform for a heatmap showing the walker's z height against xy position"""
    return env.unwrapped.state.get_walker_position().copy()

def walker_xy_timestep_transform(env, timestep, **kwargs):
    """Transform for a heatmap showing at what timestep the walker reached position xy"""
    pos = walker_xyz_transform(env)
    pos[2] = timestep
    return pos

def walker_xy_reward_transform(env, rew, **kwargs):
    """Transform for a heatmap showing what reward the walker received at position xy"""
    pos = walker_xyz_transform(env)
    pos[2] = rew
    return pos

def walker_xy_speed_transform(env, **kwargs):
    """Transform for a heatmap showing the walker's speed at position xy"""
    xy_vel = env.unwrapped.state.get_walker_velocity()[:2]
    pos = walker_xyz_transform(env)
    pos[2] = np.linalg.norm(xy_vel)
    return pos

def walker_xy_direction_transform(env, **kwargs):
    """Transform for a heatmap showing the walker's direction of velocity at position xy"""
    xy_vel = env.unwrapped.state.get_walker_velocity()[:2]
    pos = walker_xyz_transform(env)
    pos[2] = np.arctan2(xy_vel[1], xy_vel[0])
    return pos

def dummy_end_transform(**kwargs):
    pass

def final_step_end_transform(batch: np.ndarray, **kwargs):
    """Erases all steps except the final step. Useful for visualizing termination/truncation states"""
    n = batch.shape[0]
    batch[:n-1,0] = np.nan

def curry_returns_end_transform(gamma):
    """Curry an end_transform that takes a batch of step rewards and computes the discounted returns"""
    def returns_end_transform(batch: np.ndarray, **kwargs):
        n = batch.shape[0]
        for t in reversed(range(n-1)):
            batch[t,2] += gamma * batch[t+1,2]
    return returns_end_transform

def get_minimujo_heatmap_loggers(env: gym.Env, decay: float = 1, gamma: float = 1) -> List[HeatmapLogger]:
    loggers = []
    extent = get_minimujo_extent(env)

    loggers.append(HeatmapLogger(
        'walker_position_density_heatmap', 
        step_transform=walker_xyz_transform, 
        decay=decay,
        should_log_density=True,
        is_log_scale=True,
        axes_label=('X', 'Y', 'Density'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_termination_density_heatmap', 
        step_transform=walker_xyz_transform, 
        end_transform=final_step_end_transform,
        decay=decay,
        should_log_density=True,
        is_log_scale=True,
        axes_label=('X', 'Y', 'Density'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_timestep_heatmap', 
        step_transform=walker_xy_timestep_transform,
        axes_label=('X', 'Y', 'Timestep'),
        extent=extent,
        value_min=0
    ))

    loggers.append(HeatmapLogger(
        'walker_speed_heatmap', 
        step_transform=walker_xy_speed_transform,
        decay=decay,
        axes_label=('X', 'Y', 'Speed'),
        extent=extent,
        value_min=0
    ))

    loggers.append(HeatmapLogger(
        'walker_direction_heatmap', 
        step_transform=walker_xy_direction_transform,
        decay=decay,
        axes_label=('X', 'Y', 'Direction'),
        extent=extent,
        color_map='hsv',
        value_min=-np.pi,
        value_max=np.pi
    ))

    loggers.append(HeatmapLogger(
        'walker_reward_heatmap', 
        step_transform=walker_xy_reward_transform,
        decay=decay,
        axes_label=('X', 'Y', 'Reward'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_returns_heatmap', 
        step_transform=walker_xy_reward_transform,
        end_transform=curry_returns_end_transform(gamma),
        decay=decay,
        axes_label=('X', 'Y', 'Return'),
        extent=extent
    ))

    return loggers