import numpy as np

from grid_puzzle.grid import Grid


class HintSolver:
    def __init__(self, grid: Grid):
        self.grid = grid

    def solve(self):
        solution = {}
        for piece in self.grid.pieces:
            for coordinates, hint_piece in self.grid.hint_pieces.items():
                if np.array_equal(piece.img, hint_piece.img):
                    solution[coordinates] = piece
        return [solution]

    def get_solution_img(self, solution):
        grid_size = self.grid.size
        piece_size = self.grid.piece_size
        s_img = np.zeros((grid_size[1] * piece_size[1], grid_size[0] * piece_size[0], 3), 'uint8')
        for loc, piece in solution.items():
            s_img[loc[1] * piece_size[1]:loc[1] * piece_size[1] + piece_size[1],
            loc[0] * piece_size[0]:loc[0] * piece_size[0] + piece_size[0]] = piece.img

        return s_img
