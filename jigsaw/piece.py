import math

import cv2
import numpy as np
from scipy.stats import norm

from jigsaw.match_directions import MatchDir
from jigsaw.pieces_types import PieceType


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
        self.bottom=bottom_contour
        self.corners=corners
        self.type = type
        
    def match(self, other_piece, n_sample, direction: MatchDir, acceptable_ratio=1.0, max_error=0):
        padding = [0, 0, 0, 0]  # top bottom left right
        self_img = self.sub_img.copy()
        other_img = other_piece.sub_img.copy()
        if direction == MatchDir.RIGHT:
            if self.type == PieceType.LEFT_UP or self.type == PieceType.CENTER_UP:
                if self.h > other_piece.h:
                    other_img = cv2.copyMakeBorder(other_img, 0, self.h - other_piece.h, 0, 0, cv2.BORDER_CONSTANT,
                                                   value=(0, 0, 0))
                    padding[1] += self.h - other_piece.h
                else:
                    self_img = cv2.copyMakeBorder(self_img, 0, other_piece.h - self.h, 0, 0, cv2.BORDER_CONSTANT,
                                                  value=(0, 0, 0))
                    padding[1] += other_piece.h - self.h
            else:
                pass  # TODO
            padding[2] += self.w
            cat = cv2.hconcat([self_img, other_img])
        if direction == MatchDir.LEFT:
            cat = cv2.hconcat([other_img, self_img])
        if direction == MatchDir.DOWN:
            cat = cv2.vconcat([self_img, other_img])
        if direction == MatchDir.UP:
            cat = cv2.vconcat([other_img, self_img])

        cat_gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
        if direction == MatchDir.RIGHT or direction == MatchDir.LEFT:
            samples_y = np.linspace(0, max(self.h, other_piece.h), n_sample, False, dtype='int')
            matches = []
            for y in samples_y:
                row = cat_gray[y, :]
                row = np.trim_zeros(row)
                zeros_runs = Piece._zeros_runs(row)
                if len(zeros_runs) == 1:
                    matches.append(zeros_runs[0][1] - zeros_runs[0][0])
            max_match = max(matches)  # TODO refactor
            for i in range(len(matches)):
                matches[i] /= max_match
            mu, std = norm.fit(matches)
            error = 0
            for match in matches:
                if abs(match - mu) < std * 3:
                    error += math.pow(match - mu, 2)
            if error < max_error:
                padding[2] -= mu * max_match
                return True, padding
            else:
                return False, None


    # https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    def _zeros_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    def sift_features(self):
        gray = cv2.cvtColor(self.sub_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(gray, self.mask[self.y:self.y + self.h, self.x:self.x + self.w])
    def display_with_contours(self):
            # Create a black image
            display_img = np.zeros_like(self.sub_img)
            # Draw the contours with different colors
            cv2.drawContours(display_img, [self.left_contour], -1, (0, 0, 255), 2)  # Red for left contour
            cv2.drawContours(display_img, [self.right_contour], -1, (0, 255, 0), 2)  # Green for right contour
            #cv2.drawContours(display_img, [self.top_contour], -1, (255, 0, 0), 2)  # Blue for top contour
            cv2.drawContours(display_img, [self.bottom_contour], -1, (255, 255, 0), 2)  # Cyan for bottom contour
            # Draw corners
            for corner in self.corners:
                cv2.circle(display_img, tuple(corner), 5, (0, 255, 255), -1)  # Yellow for corners
            # Display the image
            cv2.imshow("Piece with Contours and Corners", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
