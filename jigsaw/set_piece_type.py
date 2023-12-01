import cv2
import numpy as np
from jigsaw.pieces_types import PieceType
from jigsaw.piece import Piece

direction_mapping = {
    (0, 0, 0, 0): PieceType.CENTER,
    (1, 0, 0, 0): PieceType.CENTER_UP,
    (0, 1, 0, 0): PieceType.CENTER_DOWN,
    (0, 0, 1, 0): PieceType.CENTER_RIGHT,
    (0, 0, 0, 1): PieceType.CENTER_LEFT,
    (1, 0, 0, 1): PieceType.LEFT_UP,
    (1, 0, 1, 0): PieceType.RIGHT_UP,
    (0, 1, 1, 0): PieceType.RIGHT_DOWN,
    (0, 1, 0, 1): PieceType.LEFT_DOWN
}

def set_piece_type(piece):
    sub_img_bbox = piece.x, piece.y, piece.w, piece.h
    mask = piece.mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # direction_holder = [up, down, right, left]; this is the direction of the piece type
    direction_holder = [0, 0, 0, 0]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                if x2 < sub_img_bbox[0] + 10:
                    direction_holder[3] = 1
                elif x1 > sub_img_bbox[0] + sub_img_bbox[2] - 10:
                    direction_holder[2] = 1
            if y2 == y1:
                if y2 < sub_img_bbox[1] + 10:
                    direction_holder[0] = 1
                elif y1 > sub_img_bbox[1] + sub_img_bbox[3] - 10:
                    direction_holder[1] = 1
    print(direction_holder)
    piece.type = direction_mapping[tuple(direction_holder)]