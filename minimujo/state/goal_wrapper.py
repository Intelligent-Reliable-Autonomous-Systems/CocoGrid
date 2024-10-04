from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import heapq
from typing import Any, Callable, Collection, Dict, Hashable, Sequence, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np

Observation = TypeVar('Observation')
AbstractState = TypeVar('AbstractState', bound=Hashable)

class GoalWrapper(gym.Wrapper):

    def __init__(
        self, 
        env: gym.Env, 
        abstraction: Callable[[Observation, gym.Env], AbstractState], 
        goal: Union[AbstractState, Callable[[Observation, AbstractState, gym.Env], AbstractState]],
        planner: SubgoalPlanner,
        observer: GoalObserver
    ) -> None:
        super().__init__(env)
        
        if not callable(goal):
            goal = lambda obs, abstract, env: goal
        self._goal_getter = goal
        self._abstraction = abstraction
        self._planner = planner
        self._observer = observer

        base_obs_space = self.env.unwrapped.observation_space
        self.observation_space = observer.transform_observation_space(base_obs_space)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(*args, **kwargs)

        abstract_state = self._abstraction(obs, self.env)
        goal_state = self._goal_getter(obs, abstract_state, self.env)
        self._planner.set_goal(goal_state)
        self._planner.update_state(abstract_state)
        self._planner.update_plan()
        
        self._initial_plan_cost = self._planner.cost

        return np.concatenate([obs, self._observer.observe(self.goal)]), info

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        
        prev_abstract_state = self.abstract_state
        prev_subgoal = self.subgoal
        prev_cost = self._planner.cost

        abstract_state = self._abstraction(obs, self.env)
        self._planner.update_state(abstract_state)
        self._planner.update_plan()

        rew += self.extra_reward(obs, prev_abstract_state, prev_subgoal, prev_cost)

        info['goal'] = self.subgoal
        info['num_subgoals'] = self._planner.cost
        info['frac_subgoals'] = self._planner.cost / self._initial_plan_cost
        
        return np.concatenate([obs, self._observer.observe(self.subgoal)]), rew, term, trunc, info
    
    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        """Override this function to do reward shaping"""
        return 0

    @property
    def goal(self) -> AbstractState:
        return self._planner.goal
    
    @property
    def abstract_state(self) -> AbstractState:
        return self._planner.state
    
    @property
    def subgoal(self) -> AbstractState:
        return self._planner.next_state
    
class GoalObserver:

    def __init__(self, goal_observation_fn, low, high):
        self.goal_observation_fn = goal_observation_fn
        self.low = low
        self.high = high

    def transform_observation_space(self, observation_space: gym.Space):
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        new_low = np.concatenate([observation_space.low, self.low], axis=None)
        new_high = np.concatenate([observation_space.high, self.high], axis=None)
        return gym.spaces.Box(low=new_low, high=new_high, dtype=observation_space.dtype)
    
    def observe(self, goal: AbstractState) -> np.ndarray:
        return self.goal_observation_fn(goal)

class SubgoalPlanner(ABC):

    def __init__(self):
        # Both set_goal and update_state must be initialized before planning
        self._current_state: AbstractState = None
        self._goal_state: AbstractState = None

    def set_goal(self, goal: AbstractState) -> None:
        """Sets the target final goal"""
        self._goal_state = goal

    def update_state(self, state: AbstractState) -> None:
        """Updates the currently achieved subgoal"""
        self._current_state = state
    
    @property
    def goal(self) -> AbstractState:
        """Gets the target final goal"""
        return self._goal_state
    
    @property
    def state(self) -> AbstractState:
        """Gets the currently achieved subgoal"""
        return self._current_state
    
    @property
    def next_state(self) -> AbstractState:
        """Gets the next goal in the plan"""
        if len(self.plan) < 2:
            return self.state
        return self.plan[1]

    @property
    @abstractmethod
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        pass

    @property
    @abstractmethod
    def cost(self) -> float:
        """Gets the cost of the computed plan"""
        pass

    @abstractmethod
    def update_plan(self) -> None:
        """Computes or updates the plan sequence of subgoals"""
        pass

class DjikstraBackwardsPlanner(SubgoalPlanner):

    def __init__(self, neighbors_fn: Callable[[AbstractState], Collection[(AbstractState, float)]]):
        """Takes a neighbors function that gets the (neighbor, weight) for each incoming edge into a state. This allows you to search backwards from the goal"""
        super().__init__()

        self._neighbors_fn = neighbors_fn
        self._is_initialized = False
        self._plan = None
        self._plan_cost = 0

    def set_goal(self, goal: AbstractState) -> None:
        """Sets the target final goal"""
        if self._goal_state == goal:
            return
        self._is_initialized = False
        self._goal_state = goal
    
    @property
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan"
        return self._plan

    @property
    def cost(self) -> float:
        """Gets the cost of the computed plan"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan cost"
        return self._distances[self.state]

    def update_plan(self) -> None:
        # Use Djisktra
        if not self._is_initialized:
            self._init_djikstra()
        elif self.state in self._plan:
            state_idx = self._plan.index(self.state)
            self._plan = self._plan[state_idx:]
            return
        elif self.state in self._visited:
            self._plan = self._reconstruct_path(self.state)
            return

        while len(self._priority_queue) > 0:
            # Get the state with the lowest known distance
            prioritized_item = heapq.heappop(self._priority_queue)
            cur_dist, cur_state = prioritized_item.priority, prioritized_item.item

            # If we've already visited this state, skip it
            if cur_state in self._visited:
                continue

            # Mark the state as visited
            self._visited.add(cur_state)

            # If we've reached the current state, return the distance
            if cur_state == self.state:
                self._plan = self._reconstruct_path(cur_state)

            # Explore neighbors of the current state
            for neighbor, cost in self._neighbors_fn(cur_state):
                if neighbor in self._visited:
                    continue
                
                new_distance = cur_dist + cost
                # If a shorter path to the neighbor is found
                if neighbor not in self._distances or new_distance < self._distances[neighbor]:
                    self._distances[neighbor] = new_distance
                    self._predecessors[neighbor] = cur_state
                    heapq.heappush(self._priority_queue, PrioritizedItem(new_distance, neighbor))
        
        return None  # No path found
    
    def _init_djikstra(self) -> None:
        assert self.goal is not None, "Goal has not been set"
        assert self.state is not None, "Current subgoal has not been set"
        self._visited = set()
        self._distances = {self.goal: 0}
        self._priority_queue = [PrioritizedItem(0, self.goal)]
        self._predecessors = {}
        self._is_initialized = True
        self._plan = None

    def _reconstruct_path(self, state: AbstractState) -> None:
        path = []
        while state in self._predecessors:
            path.append(state)
            state = self._predecessors[state]
        path.append(self.goal)
        return path

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)