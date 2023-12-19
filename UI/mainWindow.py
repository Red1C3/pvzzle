import tkinter as tk
import tkinter.filedialog

import cv2
from PIL import Image, ImageTk

from grid_puzzle.greedy_solver import GreedySolver
from grid_puzzle.grid import Grid
from grid_puzzle.hint_quant_solver import HintQuantSolver
from grid_puzzle.hint_solver import HintSolver
from jigsaw.jigsaw import Jigsaw
from utils import img_utils


class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.winfo_toplevel().title('PVZZLE')
        self.pack()
        self.drop_down_selected = self.create_type_options_menu()
        next_button = tk.Button(
            self, text="Next", command=self.transit_to_image_selection).pack()
        self.selected_img_filename = ''
        self.shuffle_var = tk.BooleanVar(value=False)
        self.grid_width_var = tk.IntVar(value=0)
        self.grid_height_var = tk.IntVar(value=0)
        self.selected_hint_filename = ''
        self.selected_algorithm = tk.StringVar()
        # PhotoImage are garbage collected so we're preventing that here
        self.images = [None, None, None]

    def create_type_options_menu(self):
        options = [
            'Grid With Hint',
            'Grid Without Hint',
            'Jigsaw With Hint'
        ]

        clicked = tk.StringVar()
        clicked.set(options[0])

        drop_down = tk.OptionMenu(self, clicked, *options)
        drop_down.pack()
        return clicked

    def transit_to_image_selection(self):
        selection_type = self.drop_down_selected.get()
        for widget in self.winfo_children():
            widget.destroy()

        def set_selected_img_filename():
            self.selected_img_filename = tk.filedialog.askopenfilename()

        tk.Button(self, text='Select an image',
                  command=set_selected_img_filename).pack()

        if selection_type == 'Grid With Hint' or selection_type == 'Grid Without Hint':
            shuffle_button = tk.Radiobutton(
                self, text="Shuffle", variable=self.shuffle_var, indicatoron=False, value=True)
            shuffle_button.pack()
            tk.Label(self, text="Enter Grid Width:").pack()
            grid_width = tk.Entry(self, textvariable=self.grid_width_var)
            grid_width.pack()
            tk.Label(self, text='Enter Grid Height:').pack()
            grid_height = tk.Entry(self, textvariable=self.grid_height_var)
            grid_height.pack()
            if selection_type == 'Grid Without Hint':
                tk.Button(self, text='Solve', command=self.solve_grid_without_hint).pack()
            else:
                def set_selected_hint_filename():
                    self.selected_hint_filename = tk.filedialog.askopenfilename()

                tk.Button(self, text='Select a hint (Defaults to image)',
                          command=set_selected_hint_filename).pack()
                algorithms = ['SIFT', 'Color Quantization']
                self.selected_algorithm.set(algorithms[0])
                algorithms_drop_down = tk.OptionMenu(
                    self, self.selected_algorithm, *algorithms)
                algorithms_drop_down.pack()
                tk.Button(self, text='Solve',
                          command=self.solve_grid_with_hint).pack()
        else:
            def set_selected_hint_filename():
                self.selected_hint_filename = tk.filedialog.askopenfilename()

            tk.Button(self, text='Select a hint (Defaults to image)',
                      command=set_selected_hint_filename).pack()
            algorithms = ['Clustering',
                          'Grid Quantization', 'Kernel Quantization']
            self.selected_algorithm.set(algorithms[0])
            algorithms_drop_down = tk.OptionMenu(
                self, self.selected_algorithm, *algorithms)
            algorithms_drop_down.pack()
            tk.Button(self, text='Solve', command=self.solve_jigsaw_with_hint).pack()

    def solve_grid_with_hint(self):
        img_path = self.selected_img_filename
        hint_path = self.selected_hint_filename
        shuffle = self.shuffle_var.get()
        solver_type = self.selected_algorithm.get()
        size = (self.grid_width_var.get(), self.grid_height_var.get())
        img = img_utils.read_img(img_path)
        if hint_path == '':
            hint = img
        else:
            hint = img_utils.read_img(hint_path)

        grid = Grid(img, size, shuffle=shuffle, hint=hint)

        grid_img = grid.get_pieces_img()

        if solver_type == "SIFT":
            solver = HintSolver(grid)
        elif solver_type == "Color Quantization":
            solver = HintQuantSolver(grid)
        solutions = solver.solve()

        solution = solver.get_solution_img(solutions[0])

        self.build_images_ui(['Grid', 'Hint', 'Solution'], [
                             grid_img, hint, solution])

    def solve_grid_without_hint(self):
        img_path = self.selected_img_filename
        shuffle = self.shuffle_var.get()
        size = (self.grid_width_var.get(), self.grid_height_var.get())
        img = img_utils.read_img(img_path)

        grid = Grid(img, size, shuffle=shuffle, hint=None)

        grid_img = grid.get_pieces_img()

        grid.process_all_pieces()
        grid.clean_up_dicts()

        solver = GreedySolver(grid)
        solutions = solver.solve()
        solution = solver.get_solution_img(solutions[0])

        self.build_images_ui(['Grid', 'Solution'], [grid_img, solution])

    def solve_jigsaw_with_hint(self):
        img_path = self.selected_img_filename
        hint_path = self.selected_hint_filename
        solver_type = self.selected_algorithm.get()
        img = img_utils.read_img(img_path)
        hint = img_utils.read_img(hint_path)

        jigsaw = Jigsaw(img, hint)

        match solver_type:
            case 'Clustering':
                solution = jigsaw.clusters_img()
            case 'Grid Quantization':
                solution = jigsaw.grid_match(5)
            case 'Kernel Quantization':  # Takes a lot of time
                solution = jigsaw.template_match2()

        self.build_images_ui(['Jigsaw Pieces', 'Hint', 'Solution'], [img, hint, solution])

    def build_images_ui(self, labels, images):
        for widget in self.winfo_children():
            widget.destroy()
        for i in range(len(labels)):
            img = images[i]
            b, g, r = cv2.split(img)
            img = cv2.merge((r, g, b))
            img = Image.fromarray(img)
            self.images[i] = ImageTk.PhotoImage(image=img)
            tk.Label(self, image=self.images[i]).grid(column=i, row=0)
            tk.Label(self, text=labels[i]).grid(column=i, row=1)
