from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

class GridWrapper(Grid):
    def __init__(self, grid: Grid):
        self.width = grid.width
        self.height = grid.height

        self.grid = grid.grid

        self.overlaps = [ [] for _ in range(self.width * self.height) ]
        self.mock_queue = []
        self.priority = None

    def set(self, i: int, j: int, v: WorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {i} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        is_removing = v is None
        flat_idx = j * self.width + i
            # If we are removing an item, search for the priority in the overlaps
        if is_removing:
            # If there is a priority
            if self.priority and self.priority[0] == i and self.priority[1] == j:
                v = self.priority[2]
            if v in self.overlaps[flat_idx]:
                self.overlaps[flat_idx].remove(v)
            elif len(self.overlaps[flat_idx]) > 0:
                self.overlaps[flat_idx].pop()
            else:
                self.grid[flat_idx] = None
            return
        
        if not self.grid[flat_idx]:
            self.grid[flat_idx] = v
        elif v not in self.overlaps[flat_idx]:
            self.overlaps[flat_idx].append(v)
        # elif v is self.grid[flat_idx] or v in self.overlaps[flat_idx]:
        #     pass
        # else:
        #     if not self.grid[flat_idx]:
        #         self.grid[flat_idx] = v
        #     else:
        #         self.overlaps[flat_idx].append(v)

        # if self.grid[flat_idx] is None:
        #     self.grid[flat_idx] = v
        # elif v is None:
        #     if len(self.overlaps[flat_idx]) > 0:
        #         v = self.overlaps[flat_idx].pop()
        #     self.grid[flat_idx] = v
        # else:
        #     self.overlaps[flat_idx].append(v)

    def get(self, i: int, j: int) -> WorldObj | None:
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.grid is not None
        # if len(self.mock_queue) > 0:
        #     return self.mock_queue.pop(0)
        if self.priority and self.priority[0] == i and self.priority[1] == j:
            return self.priority[2]
        flat_idx = j * self.width + i
        # if len(self.overlaps[flat_idx]) > 0:
        #     return self.overlaps[flat_idx].pop()
        return self.grid[flat_idx]
    
    def get_all(self, i: int, j: int) -> list[WorldObj]:
        flat_idx = j * self.width + i
        return [self.grid[flat_idx], *self.overlaps[flat_idx]]
    
    def get_all(self, idx: int) -> list[WorldObj]:
        return [self.grid[idx], *self.overlaps[idx]]
    
    # def queue_mock_cell(self, value):
    #     self.mock_queue.append(value)
    #     self.priority = value

    def set_priority(self, i, j, value):
        self.priority = (i, j, value)

    def clear_priority(self):
        self.priority = None