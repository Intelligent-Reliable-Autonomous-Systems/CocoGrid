from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from tensorboardX import SummaryWriter

class LoggingMetric:

    def register(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int, global_step_callback: Callable = None):
        self.env = env
        self.summary_writer = summary_writer
        self.max_timesteps = max_timesteps
        if global_step_callback is None:
            global_step_callback = lambda: None
        self.global_step_callback: Callable = global_step_callback
    
    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        pass

    def on_episode_end(self, timestep: int, episode: int) -> None:
        pass

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        pass

class LoggingWrapper(gym.Wrapper):

    global_step = 0

    def __init__(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int = 1000):
        super().__init__(env)

        self.summary_writer = summary_writer
        self.metrics: List[LoggingMetric] = []

        self.episode_count = -1
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.global_step_callback = lambda: LoggingWrapper.global_step

        standard_logger = StandardLogger()
        self.subscribe_metric(standard_logger)

    def subscribe_metric(self, metric: LoggingMetric):
        self.metrics.append(metric)
        metric.register(env=self.env, summary_writer=self.summary_writer, max_timesteps=self.max_timesteps, global_step_callback=self.global_step_callback)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        self.episode_count += 1
        self.timestep = 0
        obs, info = super().reset(seed=seed, options=options)
        for metric in self.metrics:
            metric.on_episode_start(obs=obs, info=info, episode=self.episode_count)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        for metric in self.metrics:
            metric.on_step(obs=obs, rew=rew, term=term, trunc=trunc, info=info, timestep=self.timestep)
        if term or trunc:
            for metric in self.metrics:
                metric.on_episode_end(timestep=self.timestep, episode=self.episode_count)
        self.timestep += 1
        LoggingWrapper.global_step += 1
        return obs, rew, term, trunc, info
    
    def __getattr__(self, name: str) -> Any:
        if name == "subscribe_metric":
            return self.subscribe_metric
        return getattr(super(), name)
    
class StandardLogger(LoggingMetric):

    def __init__(self, label_prefix: str = 'standard') -> None:
        self.label_prefix = label_prefix
        self.episode_reward_label = f'{label_prefix}/episode_reward'
        self.average_reward_label = f'{label_prefix}/average_reward'
        self.length_reward_label = f'{label_prefix}/episode_length'

    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        self.cum_reward = 0

    def on_episode_end(self, timestep: int, episode: int) -> None:
        if self.summary_writer is not None:
            global_step = self.global_step_callback()
            self.summary_writer.add_scalar(self.episode_reward_label, self.cum_reward, global_step, summary_description="Cumulative episode reward")
            self.summary_writer.add_scalar(self.average_reward_label, self.cum_reward / timestep, global_step, summary_description="Average reward per step")
            self.summary_writer.add_scalar(self.length_reward_label, timestep, global_step, summary_description="Length of the episode")

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        self.cum_reward += rew

from minimujo.utils.logging.heatmap_logger import HeatmapLogger, get_minimujo_heatmap_loggers