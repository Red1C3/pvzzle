import numpy as np

from jigsaw import pieces_detection
from jigsaw.match_directions import MatchDir
from jigsaw.pieces_types import PieceType
from utils import img_utils

img = img_utils.read_img('./samples/lenna_jigsaw.png')

pieces = pieces_detection.extract_pieces(img)

pieces[0].type = PieceType.LEFT_UP
pieces[0].sol_x = 0
pieces[0].sol_y = 0
match, padding = pieces[0].match(pieces[1], 100, MatchDir.RIGHT, max_error=0.1)
pieces[1].sol_x = int(pieces[0].sol_x + padding[3] - padding[2])
pieces[1].sol_y = int(pieces[0].sol_y + padding[0] - padding[1])
print(pieces[1].sol_x)
print(pieces[1].sol_y)

new_img = np.zeros(img.shape, 'uint8')

img_utils.display_img(new_img)

img_utils.close_all_windows()
