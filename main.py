from jigsaw.jigsaw import Jigsaw
from utils import img_utils

img = img_utils.read_img('./samples/lenna_jigsaw.png')
hint = img_utils.read_img('./samples/lenna.png')

jigsaw = Jigsaw(img, hint)

# img_utils.display_img(jigsaw.clusters_img())
img_utils.display_img(jigsaw.grid_match(10))



img_utils.close_all_windows()
