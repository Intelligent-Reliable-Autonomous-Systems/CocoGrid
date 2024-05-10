import minigrid
import numpy as np

class MinigridGoal:

    def __init__(self, agent_pos=None):
        self.agent_pos = tuple(agent_pos)

    def __call__(self, minigrid_env, dense=False, walker_pos=None):
        if self.agent_pos is not None:
            return self.position_comparison(minigrid_env, dense, walker_pos=walker_pos)
        return 0
    
    def __str__(self):
        return f'<Subgoal: pos={self.agent_pos}>'
    
    def __repr__(self):
        return f'MinigridGoal({self.agent_pos})'
    
    def position_comparison(self, minigrid_env, dense=False, walker_pos=None):
        if int(tuple(minigrid_env.agent_pos) == self.agent_pos):
            return 1
        if dense:
            x, y = self.agent_pos
            box = x - 1/2, y - 1/2, x + 1/2, y + 1/2
            return -distance_to_box_2d(walker_pos, box)
            # return -abs(walker_pos[0] - y)
        return 0
    
def distance_to_box_2d(point, box):
    py, px = point
    x1, y1, x2, y2 = box  # Assuming box is defined as [x1, y1, x2, y2]

    # Check if the point is outside the box
    if px < x1:
        closest_x = x1
    elif px > x2:
        closest_x = x2
    else:  # Inside the box horizontally
        closest_x = px

    if py < y1:
        closest_y = y1
    elif py > y2:
        closest_y = y2
    else:  # Inside the box vertically
        closest_y = py

    distance = np.sqrt((closest_x - px) ** 2 + (closest_y - py) ** 2)
    return distance