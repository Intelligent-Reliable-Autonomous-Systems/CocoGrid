from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Any, Callable, Collection, Dict, Hashable, Sequence, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np

Observation = TypeVar('Observation')
AbstractState = TypeVar('AbstractState', bound=Hashable)
Action = TypeVar('Action')
Task = Union[AbstractState, Callable[[AbstractState], Tuple[float, bool]]]

class GoalWrapper(gym.Wrapper):

    def __init__(
        self, 
        env: gym.Env, 
        abstraction: Callable[[Observation, gym.Env], AbstractState], 
        task_getter: Callable[[Observation, AbstractState, gym.Env], Task],
        planner: SubgoalPlanner,
        observer: GoalObserver,
        use_base_reward: bool = True
    ) -> None:
        super().__init__(env)
        
        self._task_getter = task_getter
        self._abstraction = abstraction
        self._planner = planner
        self._observer = observer
        self._use_base_reward = use_base_reward

        base_obs_space = self.env.unwrapped.observation_space
        self.observation_space = observer.transform_observation_space(base_obs_space)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(*args, **kwargs)

        abstract_state = self._abstraction(obs, self.env)
        task = self._task_getter(obs, abstract_state, self.env)
        self._planner.set_task(task)
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

        info['task_reward'] = rew
        if not self._use_base_reward:
            rew = 0
        rew += self.extra_reward(obs, prev_abstract_state, prev_subgoal, prev_cost)

        info['goal'] = self.subgoal
        info['goal_achieved'] = abstract_state == prev_subgoal
        info['is_new_goal'] = prev_subgoal != self.subgoal
        info['num_subgoals'] = self._planner.cost
        info['frac_subgoals'] = self._planner.cost / self._initial_plan_cost if self._initial_plan_cost != 0 else 0
        
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
        if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
            new_low = np.concatenate([observation_space.low, self.low], axis=None)
            new_high = np.concatenate([observation_space.high, self.high], axis=None)
            return gym.spaces.Box(low=new_low, high=new_high, dtype=observation_space.dtype)
        elif isinstance(observation_space, gym.spaces.Dict):
            return observation_space
    
    def observe(self, goal: AbstractState) -> np.ndarray:
        return self.goal_observation_fn(goal)

class SubgoalPlanner(ABC):

    def __init__(self):
        # Both set_goal and update_state must be initialized before planning
        self._current_state: AbstractState = None
        self._task: Callable[[AbstractState], Tuple[float, bool]] = None
        self._goal_state: AbstractState = None

    def set_task(self, task: Task) -> None:
        """Sets the target final task, which can either be a goal state or a function that evaluates a state"""
        if not callable(task):
            self._goal_state = task
            self._task = lambda abstract: (int(abstract == task), abstract == task)
        else:
            self._goal_state = None
            self._task = task

    def update_state(self, state: AbstractState) -> None:
        """Updates the currently achieved subgoal"""
        self._current_state = state
    
    @property
    def goal(self) -> AbstractState:
        """Gets the target final goal"""
        if self._goal_state is not None:
            return self._goal_state
        # if goal was not specified, refer to the plan
        return self.plan[-1]
    
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

    def set_task(self, task: Task) -> None:
        """Sets the target final goal"""
        if self._task == task:
            return
        if callable(task):
            raise Exception("DjikstraBackwardsPlanner only supports explicit goals")
        self._goal_state = task
        self._is_initialized = False
    
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
        
        # No path found
        self.plan = [self.state]
        self._distances[self.state] = 100
        print("Could not find path", self.state, self.goal)
        return
    
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
    
class DeterministicValueIterationPlanner(SubgoalPlanner):

    def __init__(
        self, 
        actions: Collection[Action],
        transition_function: Callable[[AbstractState, Action], AbstractState],
        gamma: float = 0.95
    ):
        self.actions = actions
        self.transition_function = transition_function
        self.gamma = gamma
        self._plan = [None]
        self._values = {}
        self._convergence_threshold = (1 - self.gamma) / self.gamma * 0.1

    @property
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan"
        return self._plan
    
    @property
    def cost(self):
        return self._values[self.state]

    def update_plan(self):
        if self._plan is not None and self.state in self._plan:
            state_idx = self._plan.index(self.state)
            self._plan = self._plan[state_idx:]
            return
        elif self.state in self._values.keys():
            self._plan = self._extract_plan(self.state)
            return
        
        states = self._get_reachable_states(self.state, self._values.keys())
        rewards = {state:self._task(state)[0] for state in states}
        self._values = {state:0 for state in states}
        self._terminal = set(state for state in states if self._task(state)[1])

        has_converged = False
        while not has_converged:
            max_diff = 0
            for state in states:
                if state in self._terminal:
                    continue
                old_value = self._values[state]
                new_value = -np.inf
                for action in self.actions:
                    new_state = self.transition_function(state, action)
                    reward = rewards[new_state]
                    new_value = max(new_value, reward + self.gamma * self._values[new_state])

                max_diff = max(max_diff, abs(new_value - old_value))
                self._values[state] = new_value

            if max_diff <= self._convergence_threshold:
                has_converged = True
        
        self._plan = self._extract_plan(self.state)

    def _get_reachable_states(self, from_state: AbstractState, known_states: Collection[AbstractState]):
        reachable_states = set([from_state, *known_states])
        to_expand = deque([from_state])
        large_state_warn = len(reachable_states) + 1000
        while len(to_expand) > 0:
            state = to_expand.popleft()
            for action in self.actions:
                new_state = self.transition_function(state, action)
                if new_state not in reachable_states:
                    to_expand.append(new_state)
                reachable_states.add(new_state)
            if len(reachable_states) > large_state_warn:
                print(f"[DeterministicValueIterationPlanner] Explored {len(reachable_states)} states")
                large_state_warn += 1000
        return reachable_states

    def _extract_plan(self, start_state: AbstractState):
        plan = []
        state = start_state
        has_terminated = False
        while not has_terminated:
            plan.append(state)
            best = (None, -np.inf, False)
            for action in self.actions:
                next_state = self.transition_function(state, action)
                reward, terminate = self._task(next_state)
                next_value = reward + self.gamma * self._values[next_state]
                if next_value > best[1]:
                    best = (next_state, next_value, terminate)
            state = best[0]
            if best[2] or best[0] in plan:
                # if task terminated or a loop is created
                has_terminated = True
        plan.append(state)
        return plan
    
class AStarPlanner(SubgoalPlanner):

    def __init__(
        self,
        actions: Collection[Action],
        transition_function: Callable[[AbstractState, Action], AbstractState],
        heuristic_function: Callable[[AbstractState], float] = None
    ):
        self.actions = actions
        self.transition_function = transition_function
        self.heuristic_function = heuristic_function or (lambda abstract: 0)
        self._is_initialized = False

    def set_task(self, task: Task) -> None:
        """Sets the target final task"""
        self._is_initialized = False
        super().set_task(task)
        
    @property
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan"
        return self._plan

    @property
    def cost(self) -> float:
        """Gets the cost of the computed plan"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan cost"
        return self._costs[self.goal]
    
    def update_plan(self):
        # Use Djisktra
        if self._is_initialized and self.state in self._plan:
            state_idx = self._plan.index(self.state)
            self._plan = self._plan[state_idx:]
            return
        self._init_djikstra()

        while len(self._priority_queue) > 0:
            # Get the state with the lowest known distance
            prioritized_item = heapq.heappop(self._priority_queue)
            cur_state, g_current = prioritized_item.item

            # If we've already visited this state, skip it
            if cur_state in self._visited:
                continue

            # Mark the state as visited
            self._visited.add(cur_state)

            # Explore neighbors of the current state
            for action in self.actions:
                neighbor = self.transition_function(cur_state, action)
                if neighbor in self._visited:
                    continue
                
                edge_reward, terminate = self._task(neighbor)

                # g score is the real cost from start to neighbor
                # task costs are negative, so negate it to make costs positive (i.e. lower score is better)
                g_neighbor = g_current - edge_reward 
                # f score is heuristic cost
                f_neighbor = g_neighbor - self.heuristic_function(cur_state)
                # If a shorter path to the neighbor is found
                if neighbor not in self._costs or g_neighbor < self._costs[neighbor]:
                    self._costs[neighbor] = g_neighbor
                    self._predecessors[neighbor] = cur_state
                    if terminate:
                        self._plan = self._reconstruct_path(neighbor)
                        return
                    heapq.heappush(self._priority_queue, PrioritizedItem(f_neighbor, (neighbor, g_neighbor)))
                    
        breakpoint()
        print('failed to plan')
        
    def _init_djikstra(self) -> None:
        assert self.state is not None, "Current subgoal has not been set"
        self._visited = set()
        self._costs = {self.state: 0}
        self._priority_queue = [PrioritizedItem(-self.heuristic_function(self.state), (self.state, 0))]
        self._predecessors = {}
        self._is_initialized = True
        self._plan = None

    def _reconstruct_path(self, state: AbstractState) -> None:
        path = []
        while state in self._predecessors:
            path.append(state)
            state = self._predecessors[state]
        path.append(self.state)
        return path[::-1]

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)