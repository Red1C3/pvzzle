from utils import img_utils
from jigsaw import pieces_detection

img=img_utils.read_img('./samples/lenna_jigsaw.png')

pieces_detection.extract_pieces(img)

img_utils.close_all_windows