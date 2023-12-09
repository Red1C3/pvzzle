import numpy as np

from grid_puzzle.grid import Grid
from utils import img_utils

img = img_utils.read_img('./samples/snow.jpg')

grid = Grid(img, (4, 4))

grid.process_all_pieces()
grid.clean_up_dicts()

sol_size=len(grid.pieces)
grid_size=grid.size
solutions=[]

def solve(pieces,current_loc):
    if len(pieces)==sol_size:
        solutions.append(pieces)
        return
    if current_loc[0]!=0:
        dict=pieces[current_loc].left_dict
        for p in dict.keys():
            if p not in pieces.values():
                pieces_cpy=pieces.copy()
                pieces_cpy[(current_loc[0]-1,current_loc[1])]=p
                solve(pieces_cpy,(current_loc[0]-1,current_loc[1]))
    else:
        current_loc=(grid_size[0]-1,current_loc[1])
        dict=pieces[current_loc].up_dict
        for p in dict.keys():
            if p not in pieces.values():
                pieces_cpy=pieces.copy()
                pieces_cpy[(current_loc[0],current_loc[1]-1)]=p
                solve(pieces_cpy,(current_loc[0],current_loc[1]-1))


for piece in grid.pieces:
    solve({(grid_size[0] - 1, grid_size[1] - 1): piece}, (grid_size[0] - 1, grid_size[1] - 1))

grid_size = grid.size
piece_size = grid.piece_size
for s in solutions:
    s_img = np.zeros((grid_size[1] * piece_size[1], grid_size[0] * piece_size[0], 3), 'uint8')
    for loc, piece in s.items():
        s_img[loc[1] * piece_size[1]:loc[1] * piece_size[1] + piece_size[1],
        loc[0] * piece_size[0]:loc[0] * piece_size[0] + piece_size[0]] = piece.img

    img_utils.display_img(s_img)
