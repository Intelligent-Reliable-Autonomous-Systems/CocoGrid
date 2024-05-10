import labmaze
from labmaze import defaults as labdefaults
from minigrid.core.world_object import Wall

def get_labmaze_from_minigrid(minigrid_env):
    WALL_CHAR, EMPTY_CHAR = '*', ' '
    maze_width = minigrid_env.grid.width
    # fill in walls in flat array
    walls = [WALL_CHAR if type(s) is Wall else EMPTY_CHAR for s in minigrid_env.grid.grid]
    if minigrid_env.agent_pos is not None:
        c, r = minigrid_env.agent_pos
        walls[r * maze_width + c] = labdefaults.SPAWN_TOKEN
    # make into grid
    labmaze_matrix = [walls[i:i+maze_width] for i in range(0, len(walls), maze_width)]
    # make into string (rows separated by newlines)
    labmaze_str = '\n'.join([''.join(row) for row in labmaze_matrix]) + '\n'

    return labmaze.FixedMazeWithRandomGoals(labmaze_str)