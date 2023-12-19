import tkinter as tk

from UI.mainWindow import MainWindow
from grid_puzzle.greedy_solver import GreedySolver
from grid_puzzle.grid import Grid
from grid_puzzle.hint_quant_solver import HintQuantSolver
from grid_puzzle.hint_solver import HintSolver
from jigsaw.jigsaw import Jigsaw
from utils import img_utils


def grid_without_hint(size, img_path):
    img = img_utils.read_img(img_path)

    grid = Grid(img, size, shuffle=True, hint=None)

    img_utils.display_img(grid.get_pieces_img(), "Puzzle")

    grid.process_all_pieces()
    grid.clean_up_dicts()

    solver = GreedySolver(grid)
    solutions = solver.solve()
    img_utils.display_img(solver.get_solution_img(solutions[0]), "Solution")


def grid_with_hint(size, img_path, hint_path=None, solver_type="Q"):
    img = img_utils.read_img(img_path)
    if hint_path is None:
        hint = img
    else:
        hint = img_utils.read_img(hint_path)

    grid = Grid(img, size, shuffle=True, hint=hint)

    img_utils.display_img(grid.get_pieces_img(), "Puzzle")

    if solver_type == "N":
        solver = HintSolver(grid)
    elif solver_type == "Q":
        solver = HintQuantSolver(grid)
    solutions = solver.solve()

    for s in solutions:
        img_utils.display_img(solver.get_solution_img(s), "Solution")


def jigsaw_with_hint(img_path, hint_path, solver_type):
    img = img_utils.read_img(img_path)
    hint = img_utils.read_img(hint_path)

    img_utils.display_img(img, 'Puzzle')
    img_utils.display_img(hint, "Hint")

    jigsaw = Jigsaw(img, hint)

    match solver_type:
        case 'clusters':
            img_utils.display_img(jigsaw.clusters_img(), 'Solution')
        case 'grid_quantization':
            img_utils.display_img(jigsaw.grid_match(5), 'Solution')
        case 'bruteforce_quantization':  # Takes a lot of time
            img_utils.display_img(jigsaw.template_match2(), 'Solution')


# img_path = './samples/lenna.png'
# hint_path = './samples/lenna.png'

# grid_without_hint(size=(5,5),img_path=img_path)
# grid_with_hint(size=(5,5),img_path=img_path,hint_path=None,solver_type="Q")
# jigsaw_with_hint(img_path=img_path,hint_path=hint_path,solver_type='clusters')


# img_utils.close_all_windows()


root = tk.Tk()
main_window = MainWindow(root)
main_window.mainloop()
