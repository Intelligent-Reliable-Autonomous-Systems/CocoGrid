

from minimujo.state.grid_abstraction import GridAbstraction
from minimujo.state.minimujo_state import MinimujoState


def grid_goal_task(prev_state: MinimujoState, cur_state: MinimujoState):
    grid_state = GridAbstraction.from_minimujo_state(cur_state)
    cell_value = grid_state.walker_grid_cell
    if cell_value == GridAbstraction.GOAL_IDX:
        return 1, True
    if cell_value == GridAbstraction.LAVA_IDX:
        return -1, True
    return 0, False