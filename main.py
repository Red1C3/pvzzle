from grid_puzzle.grid import Grid
from grid_puzzle.hint_solver import HintSolver
from utils import img_utils

img = img_utils.read_img('./samples/christmas-cats-500x204.jpg')
hint = img_utils.read_img('./samples/christmas-cats-1000x408.jpg')

grid = Grid(img, (5, 5), shuffle=True, hint=hint)

img_utils.display_img(grid.get_pieces_img())

solver = HintSolver(grid)
solutions = solver.solve()

for s in solutions:
    img_utils.display_img(solver.get_solution_img(s))
