from typing import Callable
import gymnasium as gym
import numpy as np

from minimujo.state.tasks import DEFAULT_TASK_REGISTRY, infer_task

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
        if options is not None and options.get('eval', False):
            return self.eval_dist.sample_task(seed, options)
        return self.train_dist.sample_task(seed, options)

class MultiTaskEnv(gym.Wrapper):
    """A wrapper to dynamically swap between environments each episode. Be careful using .unwrapped; it will return the currently selected environment."""

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
    
def register_multitask_minigrid():
    DEFAULT_TASK_REGISTRY[MultiTaskEnv] = infer_task

    def multi_goal(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env('MiniGrid-Empty-5x5-v0', size=size) \
            .add_env('MiniGrid-RandomCorner-v0', size=size) \
            .add_env('MiniGrid-HallwayChoice-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N1-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N3-v0', size=size)
        return multitask.build_env()
    
    def multi_goal_eval_easier(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_eval('MiniGrid-Empty-5x5-v0', size=size) \
            .add_eval('MiniGrid-RandomCorner-v0', size=size) \
            .add_eval('MiniGrid-HallwayChoice-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N1-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N3-v0', size=size)
        return multitask.build_env()
    
    def multi_goal_eval_harder(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env('MiniGrid-Empty-5x5-v0', size=size) \
            .add_env('MiniGrid-RandomCorner-v0', size=size) \
            .add_env('MiniGrid-HallwayChoice-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N1-v0', size=size) \
            .add_eval('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_eval('MiniGrid-SimpleCrossingS9N3-v0', size=size)
        return multitask.build_env()
    
    def multi_goal_eval_hard1(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
            .add_env('MiniGrid-Empty-5x5-v0', size=size) \
            .add_env('MiniGrid-RandomCorner-v0', size=size) \
            .add_env('MiniGrid-HallwayChoice-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N1-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N3-v0', size=size) \
            .add_eval('MiniGrid-SimpleCrossingS9N2-v0', size=size) \
            .add_eval('MiniGrid-SimpleCrossingS9N3-v0', size=size)
        return multitask.build_env()
    
    def multi_goal_eval_hard2(size=7, **kwargs):
        multitask = MultiTaskBuilder() \
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

    gym.register(
        id='MiniGrid-MultiGoal-EvalEasier-7x7-v0',
        entry_point=lambda: multi_goal_eval_easier(size=7)
    )

    gym.register(
        id='MiniGrid-MultiGoal-EvalEasier-9x9-v0',
        entry_point=lambda: multi_goal_eval_easier(size=9)
    )

    gym.register(
        id='MiniGrid-MultiGoal-EvalHarder-7x7-v0',
        entry_point=lambda: multi_goal_eval_harder(size=7)
    )

    gym.register(
        id='MiniGrid-MultiGoal-TrainAll-EvalHard-7x7-v0',
        entry_point=lambda: multi_goal_eval_hard1(size=7)
    )

    gym.register(
        id='MiniGrid-MultiGoal-TrainHard-EvalHard-7x7-v0',
        entry_point=lambda: multi_goal_eval_hard2(size=7)
    )

    gym.register(
        id='MiniGrid-MultiGoal-EvalHarder-9x9-v0',
        entry_point=lambda: multi_goal_eval_harder(size=9)
    )

    gym.register(
        id='MiniGrid-MultiGoal-TrainAll-EvalHard-9x9-v0',
        entry_point=lambda: multi_goal_eval_hard1(size=9)
    )

    gym.register(
        id='MiniGrid-MultiGoal-TrainHard-EvalHard-9x9-v0',
        entry_point=lambda: multi_goal_eval_hard2(size=9)
    )

    def doorkey_crossing_generalization(size=9, num_crossings=1, obstacle_type=None, **kwargs):
        crossing_args = {
            'size': size,
            'num_crossings': num_crossings,
        }
        if obstacle_type is not None:
            crossing_args['obstacle_type'] = obstacle_type
        multitask = MultiTaskBuilder() \
            .add_env('MiniGrid-DoorKey-6x6-v0', size=size) \
            .add_env('MiniGrid-SimpleCrossingS9N2-v0', **crossing_args) \
            .add_eval('MiniGrid-DoorKeyCrossingS9N3-v0', **crossing_args)
        return multitask.build_env()

    gym.register(
        id='MiniGrid-DoorKeyCrossingGeneralizationS9N3-v0',
        entry_point=lambda: doorkey_crossing_generalization(size=9, num_crossings=3)
    )