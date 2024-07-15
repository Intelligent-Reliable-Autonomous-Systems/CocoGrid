

from typing import Any, Callable, Dict, List, Literal, Tuple

import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tensorboardX import SummaryWriter

from minimujo.minimujo_arena import MinimujoArena
from minimujo.utils.logging import LoggingMetric
from minimujo.utils.visualize.weighted_kde import WeightedKDEHeatmap

class HeatmapLogger(LoggingMetric):

    def __init__(self, label: str, transform: Callable, should_log_density: bool = False, extent: Tuple[float] = (0, 1, 0, 1), decay: float = 1, 
            axes_label: Tuple[str] = ('X', 'Y', 'Value'), color_map: str = 'plasma', is_log_scale: bool = False,
            origin_corner: Literal['lower', 'upper'] = 'lower') -> None:
        self.transform = transform
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

        self.current_episode = 0

    def register(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int, global_step_callback: Callable = None):
        super().register(env, summary_writer, max_timesteps, global_step_callback)
        self.buffer = np.zeros((max_timesteps, 3))

    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        self.current_episode = episode

    def on_episode_end(self, timestep: int, episode: int) -> None:
        self.heatmap.add_batch(self.buffer[:timestep,:2], self.buffer[:timestep,2])
        figure, _ = plt.subplots()
        logmap = self.heatmap.densitymap if self.should_log_density else self.heatmap.heatmap
        plt.imshow(logmap, origin=self.origin_corner, cmap=self.color_map, extent=self.extent, norm=self.norm)
        plt.colorbar(label=self.axes_label[2])
        plt.title(f"{self.label} (Episode {episode})")
        plt.xlabel(self.axes_label[0])
        plt.ylabel(self.axes_label[1])
        if self.summary_writer is not None:
            self.summary_writer.add_figure(self.label, figure, self.global_step_callback(), True)

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        if timestep > self.max_timesteps:
            return
        self.buffer[timestep] = np.asarray(self.transform(
            obs=obs, rew=rew, term=term, trunc=trunc, info=info, env=self.env, timestep=timestep
        ))

def get_minimujo_extent(env: gym.Env, without_border: bool = True):
    """Gets the extent of the heatmap figure, within the arena bounds"""
    border_off = int(without_border)
    arena: MinimujoArena = env.unwrapped.arena
    extent = (border_off, arena.maze_width - border_off, -(arena.maze_height - border_off), -border_off)
    print(extent)
    return extent

def walker_xyz_transform(env, **kwargs):
    """Transform for a heatmap showing the walker's z height against xy position"""
    return env.unwrapped.state.get_walker_position().copy()

def walker_xy_timestep_transform(env, timestep, **kwargs):
    """Transform for a heatmap showing at what timestep the walker reached position xy"""
    pos = walker_xyz_transform(env)
    pos[2] = timestep
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

def get_minimujo_heatmap_loggers(env: gym.Env) -> List[HeatmapLogger]:
    loggers = []
    extent = get_minimujo_extent(env)

    loggers.append(HeatmapLogger(
        'walker_position_density_heatmap', 
        transform=walker_xyz_transform, 
        should_log_density=True,
        is_log_scale=True,
        axes_label=('X', 'Y', 'Density'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_timestep_heatmap', 
        transform=walker_xy_timestep_transform,
        axes_label=('X', 'Y', 'Timestep'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_speed_heatmap', 
        transform=walker_xy_speed_transform,
        axes_label=('X', 'Y', 'Speed'),
        extent=extent
    ))

    loggers.append(HeatmapLogger(
        'walker_direction_heatmap', 
        transform=walker_xy_direction_transform,
        axes_label=('X', 'Y', 'Direction'),
        extent=extent,
        color_map='hsv'
    ))

    return loggers