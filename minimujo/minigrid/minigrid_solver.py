import minigrid
from collections import deque
from minimujo.minigrid.grid_wrapper import GridWrapper
from minigrid.core.world_object import Door, Key, Ball, Box, Goal, Wall, Lava

from minimujo.minigrid.minigrid_goals import MinigridGoal

class AgentDummy:

    def __init__(self, cur_pos, flat_idx) -> None:
        self.cur_pos = cur_pos
        self.color = ''
        self.flat_idx = flat_idx
        self.can_pickup = False


class MinigridSolver:
    object_types = set([Door, Key, Ball, Box, Goal])
    AGENT_KEY = 'agent'

    ACTION_MAP = {
        'left': 0,
        'right': 1,
        'forward': 2,
        'pickup': 3,
        'drop': 4,
        'toggle': 5,
        'done': 6
    }

    def __init__(self, minigrid) -> None:
        self.minigrid = minigrid
        self.minigrid.grid = minigrid.grid
        if not isinstance(self.minigrid.grid, GridWrapper):
            self.minigrid.grid = GridWrapper(self.minigrid.grid)
        c, r = self.minigrid.agent_pos
        agent_flat_idx = self.get_flat_idx(r, c)
        self.agent = AgentDummy(self.minigrid.agent_pos, agent_flat_idx)

    def find_connected_distances(self, start_idx):
        to_visit = deque([(start_idx, 0)])
        expanded = set([start_idx])
        grid_to_from = dict()
        grid_to_from[start_idx] = start_idx
        connected = []

        ag_col, ag_row = self.agent.cur_pos
        agent_pos = self.get_flat_idx(ag_row, ag_col)

        while len(to_visit) > 0:
            search_idx, dist = to_visit.popleft()
            grid_dir = self.get_grid_dir(grid_to_from[search_idx], search_idx)
            neighbors = self.get_neighbors(search_idx, dir=grid_dir)
            for neighbor in neighbors:
                if neighbor in expanded:
                    continue
                expanded.add(neighbor)
                grid_to_from[neighbor] = search_idx
                if neighbor == agent_pos:
                    path_back = self.get_path_back(grid_to_from, start_idx, neighbor)
                    connected.append((self.agent, dist + 1, path_back))
                # neighbor_row, neighbor_col = self.get_row_col(neighbor)
                # objects = self.minigrid.grid.get_all(neighbor) if isinstance(self.minigrid.grid, GridWrapper) else self.minigrid.grid.get(neighbor_col, neighbor_row)
                objects = self.get_world_objects(neighbor)
                is_blocking = False
                for world_obj in objects:
                    if type(world_obj) in MinigridSolver.object_types:
                        path_back = self.get_path_back(grid_to_from, start_idx, neighbor)
                        connected.append((world_obj, dist + 1, path_back))
                        if not world_obj.can_overlap():
                            is_blocking = True
                    elif isinstance(world_obj, Wall) or isinstance(world_obj, Lava):
                        is_blocking = True
                if not is_blocking:
                    to_visit.append((neighbor, dist + 1))
        return connected
    
    def get_flat_idx(self, row, col):
        return row * self.minigrid.grid.width + col
    
    def get_row_col(self, idx):
        return idx // self.minigrid.grid.width, idx % self.minigrid.grid.width
    
    def get_world_objects(self, neighbor):
        if isinstance(self.minigrid.grid, GridWrapper):
            return self.minigrid.grid.get_all(neighbor)
        else:
            neighbor_row, neighbor_col = self.get_row_col(neighbor)
            world_obj = self.minigrid.grid.get(neighbor_col, neighbor_row)
            if world_obj is None:
                return []
            return [world_obj]
    
    def get_grid_dir(self, from_idx, to_idx):
        diff = to_idx - from_idx
        if diff == 1:
            # then to_idx is to the right
            return 0
        if diff > 1:
            # to_idx is below
            return 1
        if diff == -1:
            # to_idx is to the left
            return 2
        if diff < -1 or diff == 0:
            # to_idx is above
            return 3

    def get_neighbors(self, idx, dir=0):
        width = self.minigrid.grid.width
        row, col = self.get_row_col(idx)
        neighbors = [None, None, None, None]
        if col + 1 < width:
            # right
            neighbors[0] = idx+1
        if row + 1 < self.minigrid.grid.height:
            # down
            neighbors[1] = idx+width
        if col - 1 >= 0:
            # left
            neighbors[2] = idx-1
        if row - 1 >= 0:
            # up
            neighbors[3] = idx-width
        # prioritize dir, then adjacent, and lastly the opposite dir
        prioritized = [neighbors[dir], neighbors[dir-3], neighbors[dir-1], neighbors[dir-2]]
        return [p for p in prioritized if p is not None]
    
    def get_path_back(self, grid_to_from, start_idx, end_idx):
        idx = end_idx
        path = [end_idx]
        while idx != start_idx:
            idx = grid_to_from[idx]
            path.append(idx)
            if len(path) > 1000:
                print("Path back has cycle")
                exit()
        return path
    
    def get_all_connections(self):
        objects = []
        for i in range(len(self.minigrid.grid.grid)):
            r, c = self.get_row_col(i)
            for world_obj in self.get_world_objects(i):
                if type(world_obj) in MinigridSolver.object_types:
                    # c, r = world_obj.cur_pos
                    world_obj.cur_pos = c, r
                    objects.append((self.get_flat_idx(r, c), world_obj))
        all_connections = { obj:self.find_connected_distances(idx) for idx, obj in objects}
        all_connections[self.agent] = self.find_connected_distances(self.agent.flat_idx)
        return all_connections
    
    def get_state_mapping(self, objects, start_pos):
        start_state = [start_pos, self.minigrid.carrying]
        state_mapping = {}

        for world_obj in objects:
            if type(world_obj) is Door and not world_obj.is_open :
                state_mapping[world_obj] = len(start_state) # index corresponding to locked door state
                val = 2 * int(not world_obj.is_open) + int(world_obj.is_locked)
                start_state.append(val) # Locked door

        return tuple(start_state), state_mapping
    
    # def filter_world_object_connections(self, current_state, connections):
    #     if ty
    
    def shortest_path_through_world_object_graph(self, graph, start_pos, goal_pos):
        start_state, state_mapping = self.get_state_mapping(graph.keys(), start_pos)

        STATE_IDX, DIST_IDX, PREV_IDX, PATH_IDX = 0, 1, 2, 3
        start_node = (start_state, 0, None, []) # State, Distance, Previous

        to_visit = deque([start_node])
        expanded = set(start_state)

        while len(to_visit) > 0:
            node = to_visit.popleft()
            state, total_dist, _, back_path = node
            cur_pos = state[0]

            if state[0] is goal_pos:
                # return the path and distance
                path = [(state, back_path)]
                back_node = node
                while back_node[PREV_IDX] is not None:
                    back_node = back_node[PREV_IDX]
                    path.append((back_node[STATE_IDX], back_node[PATH_IDX]))
                return path, total_dist
            
            if cur_pos not in graph:
                continue
            for mstate, step_dist, path in self.get_state_mutations(state_mapping, state, graph):
                if mstate not in expanded:
                    expanded.add(mstate)
                    to_visit.append((mstate, total_dist + step_dist, node, path))
            # for neighbor, step_dist, path in graph[cur_pos]:
            #     next_state = (neighbor, *state[1:])
            #     if next_state not in expanded:
            #         expanded.add(next_state)
            #         to_visit.append((next_state, total_dist + step_dist, node, path))

        return None

    def get_state_mutations(self, state_mapping, state, graph):
        cur_obj = state[0]
        cur_holding = state[1]
        mutations = []
        if cur_obj in state_mapping:
            idx = state_mapping[cur_obj]
            cur_val = state[idx]
            if type(cur_obj) is Door and cur_val > 0: # is closed
                if cur_val % 2 == 0 or (type(cur_holding) is Key and cur_obj.color == cur_holding.color):
                    # Can open door if not locked or holding key
                    new_state = state[:idx] + (0,) + state[idx+1:]
                    mutations.append((new_state, 1, ['toggle']))
                # Don't let closed door add other mutations
                return mutations
        if state[1] is not None:
            # Can drop object
            new_state = state[:1] + (None, ) + state[2:]
            mutations.append((new_state, 1, ['drop']))
        else:
            # Pick up object if free hand
            if cur_obj.can_pickup:
                new_state = state[:1] + (state[0],) + state[2:]
                mutations.append((new_state, 1, ['pickup']))
        for neighbor, step_dist, path in graph[cur_obj]:
            next_state = (neighbor, *state[1:])
            mutations.append((next_state, step_dist, path))
        return mutations
    
    def get_actions_from_idx_path(self, path, start_dir):
        agent_dir = start_dir
        actions = []
        for i in range(1,len(path)):
            from_pos = path[i-1]
            to_pos = path[i]
            target_dir = self.get_grid_dir(from_pos, to_pos)
            turns = [[], ['left'], ['left','left'],['right']]
            dir_offset = (agent_dir - target_dir + 4) % 4
            actions.extend(turns[dir_offset])
            agent_dir = target_dir
            actions.append('forward')
        return actions[:-1], agent_dir

    def get_actions_from_solution(self, solution):
        agent_dir = self.minigrid.agent_dir
        total_actions = []
        path_prefix = []
        for world_object, path in solution[::-1]:
            world_object = world_object[0]
            if len(path) < 2:
                total_actions.extend(path)
                continue
            partial_actions, agent_dir = self.get_actions_from_idx_path(path_prefix + path[::-1], agent_dir)
            path_prefix = path[1:2]
            # print(path[::-1], partial_actions, agent_dir, type(world_object))
            total_actions.extend(partial_actions)
            # if type(world_object) is Door:
            #     # print('door state', world_object.is_open)
            #     if not world_object.is_open:
            #         # print('toggle')
            #         total_actions.append('toggle')
        return total_actions
    
    def get_subgoals_from_solution(self, solution):
        subgoals = []
        path_prefix = []
        for i in range(len(solution)-1,-1,-1):
            world_object, path = solution[i]
            world_object = world_object[0]
            if len(path) < 2:
                continue
            if i > 0:
                path = path[0:]
            for agent_idx in path[::-1]:
                target_pos = agent_idx % self.minigrid.grid.width, agent_idx // self.minigrid.grid.height
                goal_func = MinigridGoal(target_pos)
                subgoals.append(goal_func)
            
        if len(subgoals) > 0:
            subgoals.pop(0)
        return subgoals
    
    def get_solution(self):
        c, r = self.minigrid.agent_pos
        agent_flat_idx = self.get_flat_idx(r, c)
        self.agent = AgentDummy(self.minigrid.agent_pos, agent_flat_idx)

        connected = self.get_all_connections()
        # print(str(minigrid_env))
        # print(connected)
        # solver.to_graphviz(connected)
        goal_obj = list(connected.keys())[0]
        for world_obj in connected.keys():
            if type(world_obj) is Ball or Goal:
                goal_obj = world_obj
                break

        return self.shortest_path_through_world_object_graph(connected, self.agent, goal_obj)

    def get_solution_actions_and_subgoals(self):
        solution_return = self.get_solution()
        if solution_return is None:
            return [], [], float('inf')
        solution, distance = solution_return
        actions = self.get_actions_from_solution(solution)
        subgoals = self.get_subgoals_from_solution(solution)
        # if type(goal_obj) is Goal:
        #     actions.append('forward')
        # elif type(goal_obj) is Ball:
        #     actions.append('pickup')
        actions.append('forward')
        return actions, subgoals, distance

    
    def get_pretty_obj_name(self, world_obj):
        cls_name = world_obj.__class__.__name__
        color = world_obj.color
        col, row = world_obj.cur_pos if world_obj.cur_pos else ('N', 'A')
        return f'{cls_name}_{color}_{col}_{row}'
    
    def to_graphviz(self, connections):
        import pygraphviz as pgv

        dot = pgv.AGraph(strict=False, directed=False)
        # mapping = {obj:str(idx) for idx, obj in enumerate(connections.keys()) }
        # for obj, name in mapping.items():
        #     dot.node(str(obj))
        node_set = set()
        for from_obj, to_objs in connections.items():
            node_set.add(from_obj)
            for to_obj in to_objs:
                node_set.add(to_obj[0])

        for node in node_set:
            dot.add_node(self.get_pretty_obj_name(node), color=node.color, pos=f'{node.cur_pos[0]},{-node.cur_pos[1]}')
        
        used_objects = set()
        for from_obj, to_objs in connections.items():
            used_objects.add(from_obj)
            for to_obj in to_objs:
                if to_obj[0] in used_objects:
                    # print('skipped', to_obj)
                    continue
                dot.add_edge(self.get_pretty_obj_name(from_obj), self.get_pretty_obj_name(to_obj[0]), len=to_obj[1], xlabel=str(to_obj[1]))

        # dot.render('minigrid-connections.')
        dot.draw('minigrid-connections.png', prog='neato')

    
if __name__ == "__main__":
    import gymnasium as gym
    minigrid_id = "MiniGrid-Empty-5x5-v0"
    # minigrid_id = "MiniGrid-Playground-v0"
    minigrid_env = gym.make(minigrid_id, render_mode='human').unwrapped
    minigrid_env.reset()
    solver = MinigridSolver(minigrid_env)
    # col, row = tuple(minigrid_env.agent_pos)
    # agent_idx = row * solver.grid.width + col
    # connected = solver.find_connected_distances(idx)

    max_steps = 50
    while max_steps > 0:
        actions, subgoals = solver.get_solution_actions_and_subgoals()
        col, row = tuple(minigrid_env.agent_pos)
        agent_idx = row * solver.minigrid.grid.width + col
        
        action_id = MinigridSolver.ACTION_MAP[actions[0]]
        _, _, terminated, truncated, _ = minigrid_env.step(action_id)

        for idx, subgoal in enumerate(subgoals):
            goal_rew = subgoal(minigrid_env)
            if goal_rew > 0:
                if idx == 0:
                    if len(subgoals) == 1:
                        print(subgoal, goal_rew)
                    else:
                        print(subgoal, goal_rew, 'next:', subgoals[idx+1])
                else:
                    print(f'skipped to idx {idx}', subgoal)

        if truncated:
            print('Truncated: did not complete environment')
            break
        if terminated:
            print('Terminated: reached terminal state')
            break
        max_steps -= 1
    # for action in actions:
    #     action_id = MinigridSolver.ACTION_MAP[action]
    #     minigrid_env.step(action_id)
    print('remaining subgoals:', subgoals)