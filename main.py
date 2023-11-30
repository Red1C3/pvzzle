from jigsaw import pieces_detection
from jigsaw.match_directions import MatchDir
from jigsaw.pieces_types import PieceType
from utils import img_utils

img = img_utils.read_img('./samples/lenna_jigsaw.png')

pieces=pieces_detection.extract_pieces(img)

pieces[0].type = PieceType.LEFT_UP
pieces[0].match(pieces[1], 100, MatchDir.RIGHT)


img_utils.close_all_windows()
