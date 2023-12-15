import numpy as np

from grid_puzzle.grid import Grid

DISTANCE_THRESHOLD = 1e-9
ERROR_THRESHOLD = 0.001


class HintSolver:
    def __init__(self, grid: Grid):
        self.grid = grid

    def sift_equal(self, piece1, piece2):
        kps1, des1 = piece1.get_sift_features()
        kps2, des2 = piece2.get_sift_features()
        error = 0
        if des1 is None and des2 is None:
            return (piece1.img[0, 0, :] == piece2.img[0, 0, :]).all()
        if (des1 is None and des2 is not None) or (des2 is None and des1 is not None):
            return False
        for de1, de2 in zip(des1, des2):
            error += np.linalg.norm(de1 - de2) ** 2
        return error / len(des1) < ERROR_THRESHOLD

    def solve(self):
        solution = {}
        for piece in self.grid.pieces:
            for coordinates, hint_piece in self.grid.hint_pieces.items():
                if self.sift_equal(piece, hint_piece):
                    solution[coordinates] = piece
        return [solution]

    def get_solution_img(self, solution):
        grid_size = self.grid.size
        piece_size = self.grid.piece_size
        s_img = np.zeros(
            (grid_size[1] * piece_size[1], grid_size[0] * piece_size[0], 3), 'uint8')
        for loc, piece in solution.items():
            s_img[loc[1] * piece_size[1]:loc[1] * piece_size[1] + piece_size[1],
                  loc[0] * piece_size[0]:loc[0] * piece_size[0] + piece_size[0]] = piece.img

        return s_img
