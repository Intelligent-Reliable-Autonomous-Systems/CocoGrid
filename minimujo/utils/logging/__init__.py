from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from tensorboardX import SummaryWriter

def capped_cubic_logging_schedule(episode_id: int) -> bool:
    """This function will trigger logging at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Useful for logging metrics that are heavy duty, like images and figures.

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class LoggingMetric:

    def register(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int, global_step_callback: Optional[Callable] = None):
        self.env = env
        self.summary_writer = summary_writer
        self.max_timesteps = max_timesteps
        if global_step_callback is None:
            global_step_callback = lambda: None
        self.global_step_callback: Callable = global_step_callback
    
    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        pass

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        pass

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        pass

class LoggingWrapper(gym.Wrapper):

    global_step = 0

    def __init__(self, env: gym.Env, summary_writer: SummaryWriter, max_timesteps: int = 1000, standard_label: str = 'standard'):
        super().__init__(env)

        self.summary_writer = summary_writer
        self.metrics: List[LoggingMetric] = []

        self.episode_count = -1
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.global_step_callback = lambda: LoggingWrapper.global_step
        self.has_logged_episode = True

        standard_logger = StandardLogger(standard_label)
        self.subscribe_metric(standard_logger)

    def subscribe_metric(self, metric: LoggingMetric):
        self.metrics.append(metric)
        metric.register(env=self.env, summary_writer=self.summary_writer, max_timesteps=self.max_timesteps, global_step_callback=self.global_step_callback)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        if not self.has_logged_episode:
            self.has_logged_episode = True
            for metric in self.metrics:
                try:
                    metric.on_episode_end(timesteps=self.timestep, episode=self.episode_count)
                except:
                    # logging should never break training
                    pass
        self.episode_count += 1
        self.timestep = 0
        obs, info = super().reset(seed=seed, options=options)
        for metric in self.metrics:
            try:
                metric.on_episode_start(obs=obs, info=info, episode=self.episode_count)
            except:
                # logging should never break training
                pass
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        for metric in self.metrics:
            try:
                metric.on_step(obs=obs, rew=rew, term=term, trunc=trunc, info=info, timestep=self.timestep)
            except:
                # logging should never break training
                pass
        self.timestep += 1
        self.has_logged_episode = False
        if term or trunc:
            self.has_logged_episode = True
            for metric in self.metrics:
                try:
                    metric.on_episode_end(timesteps=self.timestep, episode=self.episode_count)
                except:
                    # logging should never break training
                    pass
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

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        if self.summary_writer is not None:
            global_step = self.global_step_callback()
            self.summary_writer.add_scalar(self.episode_reward_label, self.cum_reward, global_step)
            self.summary_writer.add_scalar(self.average_reward_label, self.cum_reward / timesteps, global_step)
            self.summary_writer.add_scalar(self.length_reward_label, timesteps, global_step)

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        self.cum_reward += rew

class MinimujoLogger(LoggingMetric):

    def __init__(self, label_prefix: str = 'standard') -> None:
        self.label_prefix = label_prefix
        self.task_reward_label = f'{label_prefix}/task_reward'
        self.task_terminated_label = f'{label_prefix}/task_terminated'

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        if self.env is None:
            return
        task = self.env.unwrapped._task
        if self.summary_writer is not None:
            global_step = self.global_step_callback()
            self.summary_writer.add_scalar(self.task_reward_label, task.reward_total, global_step)
            self.summary_writer.add_scalar(self.task_terminated_label, task.terminated, global_step)

from minimujo.utils.logging.heatmap_logger import HeatmapLogger, get_minimujo_heatmap_loggers