from typing import Callable

import gymnasium as gym
import numpy as np


class TaskDistribution:

    def __init__(self, tasks, weights):
        self.tasks = tasks
        weights = np.asarray(weights)
        self.weights = weights / np.sum(weights)

    def sample_task(self, seed, options):
        task = np.random.choice(self.tasks, p=self.weights)
        if isinstance(task, gym.Env):
            return task
        return task(seed, options)


class TrainEvalDistribution(TaskDistribution):

    def __init__(self, train_dist: TaskDistribution, eval_dist: TaskDistribution):
        self.train_dist = train_dist
        self.eval_dist = eval_dist

    def sample_task(self, seed, options):
        if options is not None and options.get("eval", False):
            return self.eval_dist.sample_task(seed, options)
        return self.train_dist.sample_task(seed, options)


class MultiTaskEnv(gym.Wrapper):
    """A wrapper to dynamically swap between environments each episode.
    Be careful using .unwrapped; it will return the currently selected environment.
    """

    def __init__(self, task_distribution: TaskDistribution):
        self._task_distribution = task_distribution
        env, _ = self._task_distribution.sample_task(None, {})
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        self.env, reset = self._task_distribution.sample_task(seed, options)
        return reset

class MultiTaskBuilder:

    def __init__(self):
        self.tasks = []
        self.weights = []
        self.eval_tasks = []
        self.eval_weights = []

    def add_env(self, env_id: str, weight: float=1, task_seed=None, task_options=None, **env_kwargs):
        def new_task(seed, options):
            if isinstance(task_seed, Callable):
                reset_seed = task_seed()
            else:
                reset_seed = task_seed
            env = gym.make(env_id, **env_kwargs).unwrapped
            reset = env.reset(seed=reset_seed, options=task_options)
            return env, reset
        self.tasks.append(new_task)
        self.weights.append(weight)
        return self

    def add_eval(self, env_id: str, weight: float=1, task_seed=None, task_options=None, **env_kwargs):
        def new_task(seed, options):
            if isinstance(task_seed, Callable):
                reset_seed = task_seed()
            else:
                reset_seed = task_seed
            env = gym.make(env_id, **env_kwargs).unwrapped
            reset = env.reset(seed=reset_seed, options=task_options)
            return env, reset
        self.eval_tasks.append(new_task)
        self.eval_weights.append(weight)
        return self

    def build(self):
        if len(self.eval_tasks) > 0:
            train_dist = TaskDistribution(self.tasks, self.weights)
            eval_dist = TaskDistribution(self.eval_tasks, self.eval_weights)
            return TrainEvalDistribution(train_dist, eval_dist)
        return TaskDistribution(self.tasks, self.weights)

    def build_env(self):
        return MultiTaskEnv(self.build())
