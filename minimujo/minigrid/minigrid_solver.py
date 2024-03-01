import minigrid
from collections import deque
from minimujo.minigrid.grid_wrapper import GridWrapper
from minigrid.core.world_object import Door, Key, Ball, Box, Goal, Wall

class AgentDummy:

    def __init__(self, cur_pos) -> None:
        self.cur_pos = cur_pos
        self.color = ''


class MinigridSolver:
    object_types = set([Door, Key, Ball, Box, Goal])
    AGENT_KEY = 'agent'

    def __init__(self, minigrid) -> None:
        self.minigrid = minigrid
        self.grid = minigrid.grid
        if not isinstance(self.grid, GridWrapper):
            self.grid = GridWrapper(self.grid)
            self.minigrid.grid = self.grid
        self.agent = AgentDummy(self.minigrid.agent_pos)

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
        return all_connections
    
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
    solver.to_graphviz(connected)