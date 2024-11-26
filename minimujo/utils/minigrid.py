import labmaze
from labmaze import defaults as labdefaults
import minigrid.manual_control
from minigrid.core.actions import Actions
from minigrid.core.world_object import Wall, Lava

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

def get_door_direction(minigrid_env, door_x, door_y):
    offsets = [
        (0, -1), # up
        (-1, 0), # left
        (0, 1), # down
        (1, 0) # right
    ]
    grid = minigrid_env.grid
    for tile_type in [Wall, Lava]: # prioritize orientation based on walls
        for idx, (x_off, y_off) in enumerate(offsets):
            x, y = door_x + x_off, door_y + y_off
            if 0 <= x < grid.width and 0 <= y < grid.height:
                if type(grid.get(x, y)) is tile_type:
                    return idx
    return 0 # default to up

class ManualControl(minigrid.manual_control.ManualControl):

    def key_handler(self, event):
        key: str = event.key

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "w": Actions.pickup,
            "pagedown": Actions.drop,
            "s": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
            "d": Actions.done
        }
        action = key_to_action.get(key, "NaN")
        action_name = Actions(action).name if isinstance(action, int) else "no action"
        print(f'Pressed {key} -> {action_name} ({action})')
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)