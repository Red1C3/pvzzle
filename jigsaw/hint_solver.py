from jigsaw.jigsaw import Jigsaw
import numpy as np
import cv2


class HintSolver:
    def __init__(self, jigsaw: Jigsaw):
        self.jigsaw = jigsaw

    def sift_equal(self, piece1, piece2):
        kps1, des1 = piece1.get_sift_features()
        kps2, des2 = piece2.get_sift_features()
        if des1 is None and des2 is None:
            if np.all(piece1.img[0, 0, :] == piece2.img[0, 0, :]):
                return np.inf
        if (des1 is None and des2 is not None) or (des2 is None and des1 is not None):
            return 0

        matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(des1, des2, 2)
        ratio_threshold = 0.7
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        return len(good_matches)

    def solve(self):
        solution = {}
        for piece in self.jigsaw.pieces:
            piece_matches = {}
            for coordinates, hint_piece in self.jigsaw.hint_pieces.items():
                piece_matches[coordinates] = self.sift_equal(piece, hint_piece)
            piece_matches = {k: v for k, v in sorted(
                piece_matches.items(), key=lambda item: item[1], reverse=True)}
            solution[list(piece_matches.keys())[0]] = piece

        return [solution]
