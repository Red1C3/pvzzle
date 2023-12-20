import cv2
import copy
import numpy as np
import math
from enum import Enum
class PieceType(Enum):
    UNKNOWN=-1
    LEFT_UP=0
    RIGHT_UP=1
    LEFT_DOWN=2
    RIGHT_DOWN=3
    CENTER_UP=4
    CENTER_DOWN=5
    CENTER_LEFT=6
    CENTER_RIGHT=7
    CENTER=8
class Piece:
    def __init__(self, x, y, w, h, sub_img, mask, contour,left_contour=[],right_contour=[],top_contour=[],bottom_contour=[],corners=[], type: PieceType = PieceType.UNKNOWN):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.sub_img = sub_img
        self.mask = mask
        self.contour = contour
        self.left_contour=left_contour
        self.right_contour=right_contour
        self.top_contour=top_contour
        self.bottom_contour=bottom_contour
        self.corners=corners
        self.type = type
    def display_with_contours(self):
        display_img = np.zeros_like(self.sub_img)
        cv2.drawContours(display_img, [self.left_contour], -1, (0, 0, 255), 2)  
        cv2.drawContours(display_img, [self.right_contour], -1, (0, 255, 0), 2) 
        cv2.drawContours(display_img, [self.top_contour], -1, (255, 0, 0), 2)  
        cv2.drawContours(display_img, [self.bottom_contour], -1, (255, 255, 0), 2)  
        for corner in self.corners:
            cv2.circle(display_img, tuple(corner), 5, (0, 255, 255), -1)  
        cv2.imshow("Piece with Contours and Corners", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def get_quantized_img(self, levels=8):
        quantized = np.round(self.sub_img * (levels / 255)) * (255 / levels)
        return quantized, levels
    def get_quantization_vector(self, levels=8):
        qimg, _ = self.get_quantized_img(levels)
        steps = np.linspace(0, 255, levels + 1)
        vec = []
        unique, counts = np.unique(qimg[:, :, 0], return_counts=True)
        unique_counts_b = dict(zip(unique, counts))
        unique, counts = np.unique(qimg[:, :, 1], return_counts=True)
        unique_counts_g = dict(zip(unique, counts))
        unique, counts = np.unique(qimg[:, :, 2], return_counts=True)
        unique_counts_r = dict(zip(unique, counts))
        total_elements_count = qimg.shape[0] * qimg.shape[1]
        for step in steps:
            if step in unique_counts_b.keys():
                vec.append(unique_counts_b[step] / total_elements_count)
            else:
                vec.append(0)
        for step in steps:
            if step in unique_counts_g.keys():
                vec.append(unique_counts_g[step] / total_elements_count)
            else:
                vec.append(0)
        for step in steps:
            if step in unique_counts_r.keys():
                vec.append(unique_counts_r[step] / total_elements_count)
            else:
                vec.append(0)
        return vec
