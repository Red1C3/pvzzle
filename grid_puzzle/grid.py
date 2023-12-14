import random

import cv2
import numpy as np

from grid_puzzle.piece import Piece
from utils import img_utils
from scipy.stats.stats import pearsonr   
 

# Sizes are defined (width,height)
class Grid:
    def __init__(self, img_colored, size, shuffle=True, hint=None):
        self.img = img_colored
        self.size = size
        self.piece_size = ((img_colored.shape[1]) // size[0], (img_colored.shape[0]) // size[1])
        self.pieces = []
        for i in range(size[0]):
            for j in range(size[1]):
                self.pieces.append(Piece(self.get_piece(img_colored, (i, j))))
        if shuffle:
            random.shuffle(self.pieces)
        if hint is not None:
            self.hint_pieces = {}
            for i in range(size[0]):
                for j in range(size[1]):
                    self.hint_pieces[(i, j)] = Piece(self.get_piece(hint, (i, j)))

    def get_piece(self, img, coordinates):
        c = coordinates
        return img[c[1] * self.piece_size[1]:c[1] * self.piece_size[1] + self.piece_size[1],
               c[0] * self.piece_size[0]:c[0] * self.piece_size[0] + self.piece_size[0]]

    def get_window(self, two_pieces, vertical=True, window_ratio=0.1):
        if vertical:
            concated_pieces = cv2.vconcat(two_pieces)
            concated_pieces_center_vertically = two_pieces[0].shape[0]
            window_size = int(two_pieces[0].shape[0] * window_ratio)
            return concated_pieces[
                   concated_pieces_center_vertically - window_size:concated_pieces_center_vertically + window_size, :]
        else:
            concated_pieces = cv2.hconcat(two_pieces)
            concated_pieces_center_horizontally = two_pieces[0].shape[1]
            window_size = int(two_pieces[0].shape[1] * window_ratio)
            return concated_pieces[:,
                   concated_pieces_center_horizontally - window_size:concated_pieces_center_horizontally + window_size]

    def get_sobel_window_score(self, sobel_window):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sobel_window = cv2.filter2D(sobel_window, -1, kernel)  # Sharpen the sobel window
        window_vertical_center = sobel_window.shape[0] // 2
        lines_of_interest = sobel_window[window_vertical_center - 1:window_vertical_center + 1, :]
        unique_lines_of_interest = np.sort(np.unique(lines_of_interest))
        if unique_lines_of_interest[-1] == 0:
            return 0
        lines_of_interest = lines_of_interest // unique_lines_of_interest[-1]
        return np.count_nonzero(lines_of_interest)

    def get_most_popular_color_window_score(self, window):
        window_vertical_center = window.shape[0] // 2
        window_top = cv2.cvtColor(window[:window_vertical_center], cv2.COLOR_BGR2GRAY)
        window_down = cv2.cvtColor(window[window_vertical_center:], cv2.COLOR_BGR2GRAY)
        window_top = np.round(window_top * (2 / 255)) * (255 / 2)  # Quantizes into 8 uniform colors
        window_down = np.round(window_down * (2 / 255)) * (255 / 2)  # Quantizes into 8 uniform colors
        window_top = np.ravel(window_top).astype('uint8')
        window_down = np.ravel(window_down).astype('uint8')
        window_top_most_freq_color = np.argmax(np.bincount(window_top))
        window_down_most_freq_color = np.argmax(np.bincount(window_down))
        return abs(window_top_most_freq_color - window_down_most_freq_color)

    def corr(self,win,v=True):
        if v:
            center = win.shape[0]//2
            f = win[center,:]
            s = win[-1,:]

        else:
            center = win.shape[1]//2
            f = win[:,center-1]
            s = win[:,-1]

        img_utils.display_img(win)
        score = 0
        for i in range (3):
            fr = f[:,i]
            sr = s[:,i]
            score += abs(pearsonr(fr,sr)[0])
        return score/3

            

    def process_piece(self, piece, window_ratio=0.1):
        for p in self.pieces:
            v_window = self.get_window([p.img, piece.img], window_ratio=window_ratio)
            print(self.corr(v_window))
            h_window = self.get_window([p.img, piece.img], False, window_ratio=window_ratio)
            print(self.corr(h_window,v=False))
            piece.up_dict[p] = self.get_sobel_window_score(img_utils.sobel_vertical_whole_img(v_window))
            piece.left_dict[p] = self.get_sobel_window_score(img_utils.sobel_vertical_whole_img(
                np.transpose(h_window, [1, 0, 2])))

    def process_all_pieces(self, window_ratio=0.1):
        for p in self.pieces:
            self.process_piece(p, window_ratio)
            p.up_dict = {k: v for k, v in sorted(p.up_dict.items(), key=lambda item: item[1])}
            p.left_dict = {k: v for k, v in sorted(p.left_dict.items(), key=lambda item: item[1])}

    def clean_up_dicts(self):
        for p in self.pieces:
            min_val = list(p.up_dict.values())[0]
            p.up_dict = {key: val for key,
            val in p.up_dict.items() if val == min_val}
            min_val = list(p.left_dict.values())[0]
            p.left_dict = {key: val for key, val in p.left_dict.items() if val == min_val}

    def get_pieces_img(self):
        grid_size = self.size
        piece_size = self.piece_size
        i = 0
        j = 0
        s_img = np.zeros((grid_size[1] * piece_size[1], grid_size[0] * piece_size[0], 3), 'uint8')
        for piece in self.pieces:
            s_img[j * piece_size[1]:j * piece_size[1] + piece_size[1],
            i * piece_size[0]:i * piece_size[0] + piece_size[0]] = piece.img
            j += 1
            if j == grid_size[1]:
                j = 0
                i += 1
        return s_img
