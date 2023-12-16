import cv2
import numpy as np

from jigsaw import pieces_detection
from jigsaw.jigsaw import Jigsaw
from jigsaw.match_directions import MatchDir
from jigsaw.pieces_types import PieceType
from utils import img_utils
from jigsaw.hint_solver import HintSolver
img = img_utils.read_img('./samples/lenna_jigsaw.png')
hint = img_utils.read_img("./samples/lenna.png")

jig = Jigsaw(img, hint)
solver = HintSolver(jig)
solution = solver.solve()[0]

for k, v in solution.items():
    img_utils.display_img(v.sub_img, str(k))

pieces = pieces_detection.extract_pieces(img)

pieces[0].type = PieceType.LEFT_UP
pieces[0].sol_x = 0
pieces[0].sol_y = 0
match, padding = pieces[0].match(pieces[1], 100, MatchDir.RIGHT, max_error=0.1)
pieces[1].sol_x = int(pieces[0].sol_x + padding[3] - padding[2])
pieces[1].sol_y = int(pieces[0].sol_y + padding[0] - padding[1])
new_img = np.zeros(img.shape, 'uint8')

new_img[0:pieces[0].sub_img.shape[0],
        0:pieces[0].sub_img.shape[1]] += pieces[0].sub_img

print(padding)
temp_img = cv2.copyMakeBorder(pieces[1].sub_img, padding[0], padding[1], int(
    padding[2]), int(padding[3]), cv2.BORDER_CONSTANT, value=(0, 0, 0))

new_img[0:temp_img.shape[0], 0:temp_img.shape[1]] += temp_img

img_utils.display_img(new_img)

img_utils.close_all_windows()
