import cv2
import numpy as np
from grid_puzzle.piece import Piece
from utils import img_utils
# Sizes are defined (width,height)
class Grid:
    def __init__(self, img_colored, size):
        self.img = img_colored
        self.size = size
        self.piece_size = ((img_colored.shape[1]) // size[0], (img_colored.shape[0]) // size[1])
        self.pieces=[]
        for i in range(size[0]):
            for j in range(size[1]):
                self.pieces.append(Piece(self.get_piece((i,j))))

    def get_piece(self, coordinates):
        c = coordinates
        return self.img[c[1] * self.piece_size[1]:c[1] * self.piece_size[1] + self.piece_size[1],
               c[0] * self.piece_size[0]:c[0] * self.piece_size[0] + self.piece_size[0]]
    def get_window(self,two_pieces,vertical=True,window_ratio=0.1):
        if vertical:
            concated_pieces=cv2.vconcat(two_pieces)
            concated_pieces_center_vertically=two_pieces[0].shape[0]
            window_size=int(two_pieces[0].shape[0]*window_ratio)
            return concated_pieces[concated_pieces_center_vertically-window_size:concated_pieces_center_vertically+window_size,:]
        else:
            concated_pieces=cv2.hconcat(two_pieces)
            concated_pieces_center_horizontally=two_pieces[0].shape[1]
            window_size=int(two_pieces[0].shape[1]*window_ratio)
            return concated_pieces[:,concated_pieces_center_horizontally-window_size:concated_pieces_center_horizontally+window_size]
    def get_sobel_window_score(self,window):
        window_vertical_center=window.shape[0]//2
        lines_of_interest=window[window_vertical_center-1:window_vertical_center+1,:]
        lines_of_interest=lines_of_interest//255
        return np.count_nonzero(lines_of_interest)
    def process_piece(self,piece,window_ratio=0.1):
        for p in self.pieces:
            v_window=self.get_window([p.img,piece.img],window_ratio=window_ratio)
            h_window=self.get_window([p.img,piece.img],False,window_ratio=window_ratio)
            piece.up_dict[p]=self.get_sobel_window_score(img_utils.sobel_vertical_whole_img(v_window))
            piece.left_dict[p]=self.get_sobel_window_score(img_utils.sobel_vertical_whole_img(np.transpose(h_window,[1,0,2]))) #TODO make sure it's transpoing correctly
