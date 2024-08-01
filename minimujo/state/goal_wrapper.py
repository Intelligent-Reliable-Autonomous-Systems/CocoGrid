from typing import Any, Dict, Optional, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from minimujo.state.grid_abstraction import GridAbstraction

class GridPositionGoalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, term_on_reach=False, dense=False) -> None:
        super().__init__(env)
        self.position_goals = {
            (2,1): (1,1),
            (3,1): (2,1),
            (3,2): (3,1),
            (3,3): (3,2),
            (2,3): (3,3),
            (1,3): (2,3)
        }
        self.goal_path = [(1,1), (2,1), (3,1), (3,2), (3,3), (2,3)]
        self.term_on_reach = term_on_reach
        self.dense = dense

        base_obs_space = self.env.unwrapped.observation_space
        assert isinstance(base_obs_space, gym.spaces.Box) and len(base_obs_space.shape) == 1
        new_low = np.concatenate([base_obs_space.low, [0,0]], axis=None)
        new_high = np.concatenate([base_obs_space.high, [5,5]], axis=None)
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(*args, **kwargs)

        abstract_state = GridAbstraction.from_minimujo_state(self.env.unwrapped.state)
        pos = abstract_state.walker_pos
        if pos in self.position_goals:
            goal = self.position_goals[pos]
        else:
            goal = (1,1)

        self.prev_pos = pos
        self.prev_goal = goal

        return np.concatenate([obs, goal]), info


    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        
        curr_state = self.env.unwrapped.state
        abstract_state = GridAbstraction.from_minimujo_state(curr_state)
        # temporarily, hard code to UMaze
        # TODO generalize planning
        pos = abstract_state.walker_pos

        # reward for subgoals
        if self.dense:
            dist = GridAbstraction.grid_distance_from_state(self.prev_goal, curr_state)
            goal_idx = self.goal_path.index(self.prev_goal) if self.prev_goal in self.goal_path else 0
            rew = -(goal_idx + dist) / len(self.goal_path)
            # print(dist, self.prev_goal, curr_state.pose[:2] / curr_state.xy_scale)
        else:
            if pos != self.prev_pos:
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