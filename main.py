from utils import img_utils
from grid_puzzle.grid import Grid
img=img_utils.read_img('./samples/christmas-cats-500x204.jpg')

grid=Grid(img,(2,2))

grid.process_all_pieces()

#sobel=img_utils.sobel_vertical_whole_img(img)

#img_utils.display_img(grid.get_sobel_window_score(sobel))
#img_utils.display_img(sobel)
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

solve({(grid_size[0]-1,grid_size[1]-1):grid.pieces[3]},(grid_size[0]-1,grid_size[1]-1))

print(solutions)
        
