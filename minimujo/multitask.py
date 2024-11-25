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

class MultiTaskEnv(gym.Wrapper):

    def __init__(self, task_distribution: TaskDistribution):
        self._task_distribution = task_distribution
        env, _ = self._task_distribution.sample_task(None, {})
        super().__init__(env)

    def reset(self, *, seed = None, options = None):
        self.env, reset = self._task_distribution.sample_task(seed, options)
        return reset
    
class MultiTaskBuilder:

    def __init__(self):
        self.tasks = []
        self.weights = []

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

    def build(self):
        return TaskDistribution(self.tasks, self.weights)
    
    def build_env(self):
        return MultiTaskEnv(self.build())
    
def register_multitask_minigrid():
    def multi_goal(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env('MiniGrid-Empty-5x5-v0', size=size) \
            .add_env('MiniGrid-RandomCorner-v0', size=size) \
            .add_env('MiniGrid-HallwayChoice-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N1-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N3-v0', size=size)
        return multitask.build_env()
    
    gym.register(
        id='MiniGrid-MultiGoal-7x7-v0',
        entry_point=lambda: multi_goal(size=7)
    )

    gym.register(
        id='MiniGrid-MultiGoal-9x9-v0',
        entry_point=lambda: multi_goal(size=9)
    )