from grid_puzzle.greedy_solver import GreedySolver
from grid_puzzle.grid import Grid
from utils import img_utils

img = img_utils.read_img('./samples/christmas-cats-500x204.jpg')

grid = Grid(img, (6, 1))

img_utils.display_img(grid.get_pieces_img())

grid.process_all_pieces()
grid.clean_up_dicts()

solver = GreedySolver(grid)
solutions = solver.solve()

for s in solutions:
    img_utils.display_img(solver.get_solution_img(s))
