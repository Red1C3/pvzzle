import tkinter as tk
import tkinter.filedialog

import cv2
from PIL import Image, ImageTk
from screeninfo import get_monitors

from jigsaw.pieces_detection import extract_pieces
from jigsaw.solving import solve_on_contours

from grid_puzzle.greedy_solver import GreedySolver
from grid_puzzle.grid import Grid
from grid_puzzle.hint_quant_solver import HintQuantSolver
from grid_puzzle.hint_solver import HintSolver
from jigsaw.jigsaw import Jigsaw
from utils import img_utils
from grid_puzzle.grid_pieces_counter import Counter


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
        self.monitor_width = get_monitors()[0].width

    def create_type_options_menu(self):
        options = [
            'Grid With Hint',
            'Grid Without Hint',
            'Jigsaw With Hint',
            'Jigsaw Without Hint'
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

        if selection_type == 'Jigsaw Without Hint':
            self.jigsaw_without_hint()
            return
        def set_selected_img_filename():
            self.selected_img_filename = tk.filedialog.askopenfilename()
            if selection_type == 'Grid With Hint' or selection_type == 'Grid Without Hint':
                counter = Counter(self.selected_img_filename)
                (row,col)=counter.main()

                grid_width.delete(0, tk.END)
                grid_width.insert(0, row)

                grid_height.delete(0, tk.END)
                grid_height.insert(0, col)

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
                tk.Button(self, text='Solve',
                          command=self.solve_grid_without_hint).pack()
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
            tk.Button(self, text='Solve',
                      command=self.solve_jigsaw_with_hint).pack()

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

        self.build_images_ui(['Jigsaw Pieces', 'Hint', 'Solution'], [
                             img, hint, solution])

    def build_images_ui(self, labels, images):
        for widget in self.winfo_children():
            widget.destroy()
        for i in range(len(labels)):
            img = images[i]
            b, g, r = cv2.split(img)
            img = cv2.merge((r, g, b))
            img = Image.fromarray(img)
            img_ratio = img.height/img.width
            img = img.resize(
                (self.monitor_width//len(images), round(img_ratio*(self.monitor_width//len(images)))))
            self.images[i] = ImageTk.PhotoImage(image=img)
            tk.Label(self, image=self.images[i]).grid(column=i, row=0)
            tk.Label(self, text=labels[i]).grid(column=i, row=1)

    def jigsaw_without_hint(self):
        self.image_button = tk.Button(self, text="Choose Image", command=self.choose_image)
        self.image_button.pack(padx=10, pady=(20, 10))  # Adjust pady for top padding

        self.img_path_label = tk.Label(self, text='Path: ')
        self.img_path_label.pack(padx=10, pady=(20, 10))  # Adjust pady for top padding

        self.bg_color_label = tk.Label(self, text='Background Color: ')
        self.bg_color_label.pack(padx=10, pady=(0, 10))  # Adjust pady for top padding

        # Add an attribute to track whether both conditions are met
        self.conditions_met = False

        # Create the processing button and set its state to DISABLED
        self.process_button = tk.Button(self, text="Process and display", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=(0, 10))
        self.pieces_len = tk.Label(self, text=' ')
        self.pieces_len.pack(padx=10, pady=(0, 10))  # Adjust pady for top padding
        # Add an attribute to store the selected color
        self.selected_color = None

    def choose_image(self):
        img_path = tk.filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if img_path:
            import cv2
            self.img_path_label.config(text='Path: ' + img_path)
            from UI.color_selector import ColorSelector
            color_selector = ColorSelector(self, img_path)
            color_selector.show_color_selector()

            # Access the selected color after the window is closed
            if color_selector.color_selected:
                self.selected_color = color_selector.color_selected
                selected_color_str = str(self.selected_color)
                self.bg_color_label.config(text='Background Color: ' + selected_color_str)

                # Set the conditions_met attribute to True
                self.conditions_met = True

                # Enable the process button
                self.process_button.config(state=tk.NORMAL)
            else:
                print("Color selection canceled.")

    def process_image(self):
        if not self.conditions_met:
            print("Conditions not met. Cannot process.")
            return
        img_path = self.img_path_label.cget("text").split(": ")[1]
        img = cv2.imread(img_path)
        bgr_selected_color = self.selected_color
        left_up_piece, right_up_piece, left_down_piece, right_down_piece,center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces,w,h = extract_pieces(img, bgr_selected_color)
        total_pieces_count = 4 + len(center_up_pieces) + len(center_down_pieces) + len(center_left_pieces) + len(center_right_pieces) + len(center_pieces)
        self.pieces_len.config(text='Image processing complete. Pieces detected : ' + str(total_pieces_count))
        solve_on_contours(left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h)
        print("Image processing complete.")
