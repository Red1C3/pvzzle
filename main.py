from grid_puzzle.grid import Grid
from grid_puzzle.hint_quant_solver import HintQuantSolver
from utils import img_utils

img = img_utils.read_img('./samples/snow.jpg')
hint = img_utils.read_img('./samples/snow-1024.jpg')

grid = Grid(img, (15, 15), shuffle=True, hint=hint)

img_utils.display_img(grid.get_pieces_img())

solver = HintQuantSolver(grid)
solutions = solver.solve()

for s in solutions:
    img_utils.display_img(solver.get_solution_img(s))
