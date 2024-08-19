

from typing import Any, Dict
from minimujo.state.goal_wrapper import GridPositionGoalWrapper
from minimujo.utils.gym_wrapper import unwrap_env
from minimujo.utils.logging import LoggingMetric


class SubgoalLogger(LoggingMetric):

    def __init__(self, label_prefix: str = 'subgoals') -> None:
        self.label_prefix = label_prefix
        self.num_left_label = f'{label_prefix}/num_goals_left'
        self.frac_left_label = f'{label_prefix}/frac_goals_left'

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        if self.summary_writer is not None:
            global_step = self.global_step_callback()
            if self.num_left is not None:
                self.summary_writer.add_scalar(self.num_left_label, self.num_left, global_step)
            if self.frac_left is not None:
                self.summary_writer.add_scalar(self.frac_left_label, self.frac_left, global_step)

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        if term or trunc:
            self.num_left = info.get('num_subgoals', None)
            self.frac_left = info.get('frac_subgoals', None)

