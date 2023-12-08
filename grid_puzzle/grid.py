import cv2
from grid_puzzle.piece import Piece

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
