from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Door, Key, Ball, Box, Goal
import numpy as np

from minimujo.minigrid.grid_wrapper import GridWrapper
from minimujo.minigrid.minigrid_solver import MinigridSolver

class MinigridManager:
    def __init__(self, minigrid, object_entities, use_subgoal_rewards=False):
        self._minigrid = minigrid
        self._objects = object_entities

        self._use_subgoal_rewards = use_subgoal_rewards
        self._solver = None
        
        self.reset()

    def reset(self):
        self._grid = GridWrapper(self._minigrid.grid)
        self._minigrid.grid = self._grid

        if self._use_subgoal_rewards:
            self._solver = MinigridSolver(self._minigrid)

        self._replan_subgoal_interval = 10
        self._replan_subgoal_count = self._replan_subgoal_interval
        self._allow_subgoal_skips = False
        self._current_subgoals = []

        self._max_subgoals = np.inf
        self._last_dist = None
        self._subgoal_dist = 0
        self._subgoal_init_dist = float('inf')

    def has_solution(self):
        _, _, dist = self._solver.get_solution_actions_and_subgoals()
        return dist < 10000

    def subgoal_rewards(self, arena, dense=False):
        if not self._use_subgoal_rewards or self._max_subgoals == 0:
            return 0
        
        current_subgoals = self._current_subgoals
        self._replan_subgoal_count += 1
        if self._replan_subgoal_count > self._replan_subgoal_interval:
            _, self._current_subgoals, self._subgoal_dist = self._solver.get_solution_actions_and_subgoals()
            self._replan_subgoal_count = 0
            
            # do not allow backtracking to farm subgoals
            self._max_subgoals = min(len(self._current_subgoals), self._max_subgoals)
            # self._current_subgoals = self._current_subgoals[-self._max_subgoals:]

        if len(current_subgoals) == 0:
            return 0
        
        goal_rew = 0
        walker_pos = arena.walker_grid_continuous_position
        if self._allow_subgoal_skips:
            for idx, subgoal in enumerate(current_subgoals):
                goal_rew = subgoal(self._minigrid)
                if goal_rew > 0:
                    if idx == 0:
                        print('Completed subgoal', subgoal)
                    else:
                        print(f'Skipped subgoal', subgoal)
        else:
            subgoal = current_subgoals[0]
            goal_rew = subgoal(self._minigrid, dense=dense, walker_pos=walker_pos)
            if goal_rew > 0:
                print('Completed subgoal', subgoal, 'next:', current_subgoals[1] if len(current_subgoals) > 1 else '')
                current_subgoals.pop(0)
                goal_rew = max(0, (self._max_subgoals - len(current_subgoals)))
                self._max_subgoals = min(len(current_subgoals), self._max_subgoals)
                self._subgoal_dist -= 1
                if dense and len(current_subgoals) == 0:
                    return (self._last_dist) / self._subgoal_init_dist
        if dense:
            total_distance = (-goal_rew) + (self._subgoal_dist - 1)
            if self._last_dist == None:
                self._last_dist = total_distance
                self._subgoal_init_dist = total_distance
                if self._subgoal_init_dist == 0:
                    print("init dist was zero", total_distance, goal_rew, self._subgoal_dist)
                    self._subgoal_init_dist = float('inf')
            diff = (self._last_dist - total_distance) / self._subgoal_init_dist
            self._last_dist = total_distance
            return diff
        return goal_rew
    
    def get_current_goal_pos(self):
        if len(self._current_subgoals) == 0:
            return (0, 0)
        return self._current_subgoals[0].agent_pos

    def sync_minigrid(self, arena):
        reward = 0
        terminated = False

        walker_pos = arena.walker_grid_position
        for door in self._objects[Door]:
            entity = door['entity']
            miniobj = door['mini_obj']
            col, row = door['mini_pos']

            if miniobj.is_locked != entity.is_locked:
                r, t = self.unlock_door(miniobj, col, row)
                reward += r
                terminated = terminated or t

            if miniobj.is_open != entity.is_open:
                r, t = self.toggle_door(miniobj, col, row)
                reward += r
                terminated = terminated or t

        for ball in self._objects[Ball]:
            r, t = self.handle_grabbable(ball, arena)
            reward += r
            terminated = terminated or t

        for box in self._objects[Box]:
            r, t = self.handle_grabbable(box, arena)
            reward += r
            terminated = terminated or t

        for key in self._objects[Key]:
            r, t = self.handle_grabbable(key, arena)
            reward += r
            terminated = terminated or t

        if not (self._minigrid.agent_pos[0] == walker_pos[0] and self._minigrid.agent_pos[1] == walker_pos[1]):
            r, t = self.move_agent(*walker_pos)
            reward += r
            terminated = terminated or t

        for goal in self._objects[Goal]:
            entity = goal['entity']
            miniobj = goal['mini_obj']
            col, row = goal['mini_pos']
            if walker_pos[0] == col and walker_pos[1] == row:
                reward += 1
                terminated = True

        return int(reward > 0), terminated

    def handle_grabbable(self, grabbable, arena):
        reward, terminated = (0, False)

        entity = grabbable['entity']
        miniobj = grabbable['mini_obj']
        mg_col, mg_row = grabbable['mini_pos']
        entity_row, entity_col = arena.world_to_minigrid_position(entity.position) if not entity.is_grabbed else (-1, -1)
        if entity_col < -1 or entity_row < -1:
            return reward, terminated

        if self._minigrid.carrying == miniobj and not entity.is_grabbed:
            # Need to drop object
            reward, terminated = self.drop_object(miniobj, entity_col, entity_row)
            grabbable['mini_pos'] = entity_col, entity_row
            return reward, terminated

        if not (mg_col == entity_col and mg_row == entity_row):
            if entity_col > -1 and entity_row > -1:
                self.intervene_object_position(miniobj, mg_col, mg_row, entity_col, entity_row)
                grabbable['mini_pos'] = entity_col, entity_row

        if self._minigrid.carrying != miniobj and entity.is_grabbed:
            # Need to pickup object
            r, t = self.pickup_object(miniobj, mg_col, mg_row)
            grabbable['mini_pos'] = (-1, -1)
            reward += r
            terminated = terminated or t

        return reward, terminated
                
    def unlock_door(self, miniobj, col, row):
        with Intervener(self._minigrid, col, row) as intervener:
            self._grid.set_priority(col, row, miniobj)
            _, reward, terminated, _, _ = self._minigrid.step(Actions.toggle)
            self._grid.clear_priority()
        return reward, terminated

    def toggle_door(self, miniobj, col, row):
        with Intervener(self._minigrid, col, row) as intervener:
            self._grid.set_priority(col, row, miniobj)
            _, reward, terminated, _, _ = self._minigrid.step(Actions.toggle)
            self._grid.clear_priority()
        return reward, terminated

    def intervene_object_position(self, miniobj, from_col, from_row, to_col, to_row):
        self._grid.set_priority(from_col, from_row, miniobj)
        self._grid.set(from_col, from_row, None)
        self._grid.clear_priority()
        # self._grid.set_priority(to_col, to_row, miniobj)
        self._grid.set(to_col, to_row, miniobj)
        miniobj.cur_pos = (to_col, to_row)

    def pickup_object(self, miniobj, from_col, from_row):
        with Intervener(self._minigrid, from_col, from_row) as intervener:
            # target_cell = self._grid.get(from_row, from_col)
            # if target_cell is not miniobj:
            #     # Make the grid wrapper return None rather than a cell that would block the movement.
            #     self._grid.queue_mock_cell(miniobj)
            self._grid.set_priority(from_col, from_row, miniobj)
            _, reward, terminated, _, _ = self._minigrid.step(Actions.pickup)
            self._grid.clear_priority()
        return reward, terminated

    def drop_object(self, miniobj, to_col, to_row):
        with Intervener(self._minigrid, to_col, to_row) as intervener:
            # target_cell = self._grid.get(to_col, to_row)
            # if target_cell is not None:
            #     # Make the grid wrapper return None rather than a cell that would block the drop.
            #     self._grid.queue_mock_cell(None)
            self._grid.set_priority(to_col, to_row, None)
            _, reward, terminated, _, _ = self._minigrid.step(Actions.drop)
            self._grid.clear_priority()
        return reward, terminated

    def move_agent(self, to_col, to_row):
        with Intervener(self._minigrid, to_col, to_row) as intervener:
            # target_cell = self._grid.get(to_col, to_row)
            # if target_cell is not None and not target_cell.can_overlap():
            #     # Make the grid wrapper return None rather than a cell that would block the movement.
            #     self._grid.queue_mock_cell(None)
            self._grid.set_priority(to_col, to_row, None)
            _, reward, terminated, _, _ = self._minigrid.step(Actions.forward)
            self._grid.clear_priority()
        return reward, terminated

class Intervener:
    def __init__(self, minigrid, fwd_col, fwd_row):
        self._minigrid = minigrid
        self._fwd_pos = np.array([fwd_col, fwd_row])
        self._dir = 0
        self._dir_to_vec_value = None
    
    def __enter__(self):
        diff_vector = self._fwd_pos - self._minigrid.agent_pos
        self._dir = self._minigrid.agent_dir
        self._dir_to_vec_value = DIR_TO_VEC[self._dir]
        DIR_TO_VEC[self._dir] = diff_vector

    def __exit__(self, *args):
        DIR_TO_VEC[self._dir] = self._dir_to_vec_value