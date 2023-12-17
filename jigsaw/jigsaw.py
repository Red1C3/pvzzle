import cv2

from jigsaw.pieces_detection import extract_pieces
from jigsaw.set_piece_type import set_piece_type


class Jigsaw:
    def __init__(self, colored_img, hint=None):
        self.pieces = extract_pieces(colored_img)
        for piece in self.pieces:
            set_piece_type(piece)
            piece.sift = piece.sift_features()
        if hint is not None:
            self.hint = hint
            sift = cv2.SIFT_create()
            self.hint_sift_features = sift.detectAndCompute(hint, None)
