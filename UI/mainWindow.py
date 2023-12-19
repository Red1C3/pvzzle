import tkinter as tk
import tkinter.filedialog


class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.drop_down_selected = self.create_type_options_menu()
        next_button = tk.Button(self, text="Next", command=self.transit_to_image_selection).pack()

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
        selected_img_filename = ''

        def set_selected_img_filename():
            nonlocal selected_img_filename
            selected_img_filename = tk.filedialog.askopenfilename()

        tk.Button(self, text='Select an image', command=set_selected_img_filename).pack()

        if selection_type == 'Grid With Hint' or selection_type == 'Grid Without Hint':
            shuffle_var = tk.BooleanVar(value=False)
            grid_width_var = tk.IntVar(value=0)
            grid_height_var = tk.IntVar(value=0)
            shuffle_button = tk.Radiobutton(self, text="Shuffle", variable=shuffle_var, indicatoron=False, value=False)
            shuffle_button.pack()
            tk.Label(self, text="Enter Grid Width:").pack()
            grid_width = tk.Entry(self, textvariable=grid_width_var)
            grid_width.pack()
            tk.Label(self, text='Enter Grid Height:').pack()
            grid_height = tk.Entry(self, textvariable=grid_height_var)
            grid_height.pack()
            if selection_type == 'Grid Without Hint':
                tk.Button(self, text='Solve').pack()
            else:
                selected_hint_filename = ''

                def set_selected_hint_filename():
                    nonlocal selected_hint_filename
                    selected_hint_filename = tk.filedialog.askopenfilename()

                tk.Button(self, text='Select a hint (Defaults to image)', command=set_selected_hint_filename).pack()
                algorithms = ['SIFT', 'Color Quantization']
                selected_algorithm = tk.StringVar()
                selected_algorithm.set(algorithms[0])
                algorithms_drop_down = tk.OptionMenu(self, selected_algorithm, *algorithms)
                algorithms_drop_down.pack()
                tk.Button(self, text='Solve').pack()
        else:
            selected_hint_filename = ''

            def set_selected_hint_filename():
                nonlocal selected_hint_filename
                selected_hint_filename = tk.filedialog.askopenfilename()

            tk.Button(self, text='Select a hint (Defaults to image)', command=set_selected_hint_filename).pack()
            algorithms = ['Clustering', 'Grid Quantization', 'Kernel Quantization']
            selected_algorithm = tk.StringVar()
            selected_algorithm.set(algorithms[0])
            algorithms_drop_down = tk.OptionMenu(self, selected_algorithm, *algorithms)
            algorithms_drop_down.pack()
            tk.Button(self, text='Solve').pack()
