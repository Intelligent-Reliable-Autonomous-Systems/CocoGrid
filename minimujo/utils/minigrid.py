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

def minigrid_tile_generator(minigrid_env, tile_type=None):
    maze_width = minigrid_env.grid.width
    maze_height = minigrid_env.grid.height
    for x in range(maze_width):
        for y in range(maze_height):
            tile = minigrid_env.grid.get(x, y)
            if tile is not None and tile_type is None or type(tile) == tile_type:
                yield x, y, tile