import numpy as np

from grid_puzzle.grid import Grid


class HintQuantSolver:
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def get_quantized_space_distance(piece1, piece2, levels=8):
        vec1 = np.array(piece1.get_quantization_vector(levels))
        vec2 = np.array(piece2.get_quantization_vector(levels))
        return np.linalg.norm(vec1 - vec2)

    def solve(self, levels=8):
        solution = {}
        for piece in self.grid.pieces:
            piece_matches = {}
            for coordinates, hint_piece in self.grid.hint_pieces.items():
                piece_matches[coordinates] = HintQuantSolver.get_quantized_space_distance(piece, hint_piece, levels)
            piece_matches = {k: v for k, v in sorted(piece_matches.items(), key=lambda item: item[1], reverse=False)}
            solution[list(piece_matches.keys())[0]] = piece

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
