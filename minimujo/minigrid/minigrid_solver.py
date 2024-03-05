import minigrid
from collections import deque
from minimujo.minigrid.grid_wrapper import GridWrapper
from minigrid.core.world_object import Door, Key, Ball, Box, Goal, Wall

class AgentDummy:

    def __init__(self, cur_pos, flat_idx) -> None:
        self.cur_pos = cur_pos
        self.color = ''
        self.flat_idx = flat_idx
        self.can_pickup = False


class MinigridSolver:
    object_types = set([Door, Key, Ball, Box, Goal])
    AGENT_KEY = 'agent'

    def __init__(self, minigrid) -> None:
        self.minigrid = minigrid
        self.grid = minigrid.grid
        if not isinstance(self.grid, GridWrapper):
            self.grid = GridWrapper(self.grid)
            self.minigrid.grid = self.grid
        c, r = self.minigrid.agent_pos
        agent_flat_idx = self.get_flat_idx(r, c)
        self.agent = AgentDummy(self.minigrid.agent_pos, agent_flat_idx)

    def find_connected_distances(self, start_idx):
        to_visit = deque([(start_idx, 0)])
        expanded = set([start_idx])
        connected = []

        ag_col, ag_row = self.agent.cur_pos
        agent_pos = self.get_flat_idx(ag_row, ag_col)

        while len(to_visit) > 0:
            search_idx, dist = to_visit.popleft()
            neighbors = self.get_neighbors(search_idx)
            for neighbor in neighbors:
                if neighbor in expanded:
                    continue
                expanded.add(neighbor)
                if neighbor == agent_pos:
                    connected.append((self.agent, dist + 1))
                objects = self.grid.get_all(neighbor)
                is_blocking = False
                for world_obj in objects:
                    if type(world_obj) in MinigridSolver.object_types:
                        connected.append((world_obj, dist + 1))
                        if not world_obj.can_overlap():
                            is_blocking = True
                    elif isinstance(world_obj, Wall):
                        is_blocking = True
                if not is_blocking:
                    to_visit.append((neighbor, dist + 1))
        return connected
    
    def get_flat_idx(self, row, col):
        return row * self.grid.width + col
    
    def get_row_col(self, idx):
        return idx // self.grid.width, idx % self.grid.width

    def get_neighbors(self, idx):
        width = self.grid.width
        row, col = self.get_row_col(idx)
        neighbors = []
        if row - 1 >= 0:
            neighbors.append(idx-width)
        if row + 1 < self.grid.height:
            neighbors.append(idx+width)
        if col - 1 >= 0:
            neighbors.append(idx-1)
        if col + 1 < width:
            neighbors.append(idx+1)
        return neighbors
    
    def get_all_connections(self):
        objects = []
        for i in range(len(self.grid.grid)):
            r, c = self.get_row_col(i)
            for world_obj in self.grid.get_all(i):
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
            if type(world_obj) is Door and world_obj.is_locked:
                state_mapping[world_obj] = len(start_state) # index corresponding to locked door state
                start_state.append(1) # Locked door

        return tuple(start_state), state_mapping
    
    def shortest_path_through_world_object_graph(self, graph, start_pos, goal_pos):
        start_state, state_mapping = self.get_state_mapping(graph.keys(), start_pos)

        STATE_IDX, DIST_IDX, PREV_IDX = 0, 1, 2
        start_node = (start_state, 0, None) # State, Distance, Previous

        to_visit = deque([start_node])
        expanded = set(start_state)

        while len(to_visit) > 0:
            node = to_visit.popleft()
            state, total_dist, _ = node
            cur_pos = state[0]

            if state[0] is goal_pos:
                # return the path and distance
                path = [state]
                back_node = node
                while back_node[PREV_IDX] is not None:
                    back_node = back_node[PREV_IDX]
                    path.append(back_node[STATE_IDX])
                return path, total_dist
            
            if cur_pos not in graph:
                continue
            for mstate in self.get_state_mutations(state_mapping, state):
                if mstate not in expanded:
                    expanded.add(mstate)
                    to_visit.append((state, total_dist + step_dist, node))
            for neighbor, step_dist in graph[cur_pos]:
                next_state = (neighbor, *state[1:])
                if next_state not in expanded:
                    expanded.add(next_state)
                    to_visit.append((next_state, total_dist + step_dist, node))

        return None

    def get_state_mutations(self, state_mapping, state):
        cur_obj = state[0]
        mutations = []
        if cur_obj in state_mapping:
            idx = state_mapping[cur_obj]
            cur_val = state[idx]
            if type(cur_obj) is Door and cur_val > 0:
                # Unlock door
                mutations.append(state[:idx] + (0,) + state[idx+1:])
        if state[1] is not None:
            # Can drop object
            mutations.append(state[:1] + (None, ) + state[2:])
        else:
            # Pick up object if free hand
            if cur_obj.can_pickup:
                mutations.append(state[:1] + (state[0],) + state[2:])
        return mutations
    
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
    # minigrid_id = "MiniGrid-KeyCorridorS3R2-v0"
    minigrid_id = "MiniGrid-Playground-v0"
    minigrid_env = gym.make(minigrid_id).unwrapped
    minigrid_env.reset()
    solver = MinigridSolver(minigrid_env)
    col, row = tuple(minigrid_env.agent_pos)
    idx = row * solver.grid.width + col
    # connected = solver.find_connected_distances(idx)
    connected = solver.get_all_connections()
    print(str(minigrid_env))
    print(connected)
    # solver.to_graphviz(connected)
    goal_obj = list(connected.keys())[0]
    solution = solver.shortest_path_through_world_object_graph(connected, solver.agent, goal_obj)
    print('solution', solution, goal_obj)