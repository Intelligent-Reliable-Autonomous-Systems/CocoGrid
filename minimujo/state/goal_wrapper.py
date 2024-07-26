from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from minimujo.state.grid_abstraction import GridAbstraction

class GridPositionGoalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, term_on_reach=False, completion_reward=False) -> None:
        super().__init__(env)
        self.position_goals = {
            (2,1): (1,1),
            (3,1): (2,1),
            (3,2): (3,1),
            (3,3): (3,2),
            (2,3): (3,3),
            (1,3): (2,3)
        }
        self.term_on_reach = term_on_reach

        base_obs_space = self.env.unwrapped.observation_space
        assert isinstance(base_obs_space, gym.spaces.Box) and len(base_obs_space.shape) == 1
        new_low = np.concatenate([base_obs_space.low, [0,0]], axis=None)
        new_high = np.concatenate([base_obs_space.high, [5,5]], axis=None)
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.prev_goal = None
        self.prev_pos = None
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        
        abstract_state = GridAbstraction.from_minimujo_state(self.env.unwrapped.state)
        # temporarily, hard code to UMaze
        # TODO generalize planning
        pos = abstract_state.walker_pos

        # reward for subgoals
        if self.prev_pos is not None and pos != self.prev_pos:
            if pos == self.prev_goal:
                rew += 1
            else:
                rew -= 1

        if pos in self.position_goals:
            goal = self.position_goals[pos]
        else:
            goal = (1,1)

        self.prev_pos = pos
        self.prev_goal = goal

        if self.term_on_reach and pos == goal:
            term = True
        
        return np.concatenate([obs, goal]), rew, term, trunc, info