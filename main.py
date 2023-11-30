import cv2.cuda

from utils import img_utils
from jigsaw import pieces_detection

img = img_utils.read_img('./samples/lenna_jigsaw.png')

pieces=pieces_detection.extract_pieces(img)


for piece in pieces:
    img_utils.display_img(piece.sub_img)

img_utils.close_all_windows()
