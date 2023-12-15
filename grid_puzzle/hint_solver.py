import numpy as np

from grid_puzzle.grid import Grid

DISTANCE_THRESHOLD = 0.01
ERROR_THRESHOLD = 100000


class HintSolver:
    def __init__(self, grid: Grid):
        self.grid = grid

    def sift_equal(self, piece1, piece2):
        kps1, des1 = piece1.get_sift_features()
        kps2, des2 = piece2.get_sift_features()
        matching_points_count = 0
        error = 0
        for kp in kps1:
            kp.pt = (
                kp.pt[0]/piece1.img.shape[1], kp.pt[1]/piece1.img.shape[0])
        for kp in kps2:
            kp.pt = (
                kp.pt[0]/piece2.img.shape[1], kp.pt[1]/piece2.img.shape[0])

        for kp1, de1 in zip(kps1, des1):
            for kp2, de2 in zip(kps2, des2):
                dis = np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
                if dis < DISTANCE_THRESHOLD:
                    matching_points_count += 1
                    error += np.linalg.norm(de1 - de2)**2
                    if error > ERROR_THRESHOLD:
                        return False
        if matching_points_count == 0:
            return False
        return error/matching_points_count < ERROR_THRESHOLD

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
