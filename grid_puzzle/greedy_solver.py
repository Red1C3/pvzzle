import numpy as np

from grid_puzzle.grid import Grid


class GreedySolver:
    def __init__(self, grid: Grid):
        self.solutions = []
        self.grid = grid

    def _solve(self, pieces, current_loc,score):
        if len(pieces) == len(self.grid.pieces):
            self.solutions.append((pieces,score))
            return
        if current_loc[0] != 0:
            solve_dict = pieces[current_loc].left_dict
            for p in solve_dict.keys():
                if p not in pieces.values():
                    pieces_cpy = pieces.copy()
                    pieces_cpy[(current_loc[0] - 1, current_loc[1])] = p
                    self._solve(pieces_cpy, (current_loc[0] - 1, current_loc[1]),score+solve_dict[p])
        else:
            current_loc = (self.grid.size[0] - 1, current_loc[1])
            solve_dict = pieces[current_loc].up_dict
            for p in solve_dict.keys():
                if p not in pieces.values():
                    pieces_cpy = pieces.copy()
                    pieces_cpy[(current_loc[0], current_loc[1] - 1)] = p
                    self._solve(pieces_cpy, (current_loc[0], current_loc[1] - 1),score+solve_dict[p])

    def solve(self):
        self.solutions = []
        grid_size = self.grid.size
        for piece in self.grid.pieces:
            self._solve({(grid_size[0] - 1, grid_size[1] - 1): piece}, (grid_size[0] - 1, grid_size[1] - 1),0)
        self.solutions = sorted(self.solutions, key=lambda item: item[1])
        return self.solutions

    def get_solution_img(self, solution):
        grid_size = self.grid.size
        piece_size = self.grid.piece_size
        s_img = np.zeros((grid_size[1] * piece_size[1], grid_size[0] * piece_size[0], 3), 'uint8')
        for loc, piece in solution[:][0].items():
            s_img[loc[1] * piece_size[1]:loc[1] * piece_size[1] + piece_size[1],
            loc[0] * piece_size[0]:loc[0] * piece_size[0] + piece_size[0]] = piece.img

        return s_img
