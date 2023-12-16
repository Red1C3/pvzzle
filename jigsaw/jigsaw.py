from grid_puzzle.grid import Grid
from grid_puzzle.piece import Piece as GPiece
from jigsaw.pieces_detection import extract_pieces
from jigsaw.pieces_types import PieceType
from jigsaw.set_piece_type import set_piece_type


class Jigsaw:
    def __init__(self, img, hint):
        self.pieces = extract_pieces(img)
        self.size = [0, 0]

        for piece in self.pieces:
            set_piece_type(piece)
            if piece.type == PieceType.LEFT_UP or piece.type == PieceType.RIGHT_UP or piece.type == PieceType.CENTER_UP:
                self.size[0] += 1
            if piece.type == PieceType.RIGHT_UP or piece.type == PieceType.RIGHT_DOWN or piece.type == PieceType.CENTER_RIGHT:
                self.size[1] += 1

        self.hint_pieces = {}
        self.hint_piece_size = ((hint.shape[1]) // self.size[0], (hint.shape[0]) // self.size[1])
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.hint_pieces[(i, j)] = GPiece(Grid.get_piece(hint, (i, j), self.hint_piece_size))
