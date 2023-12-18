import os
import tkinter as tk
from tkinter import filedialog
import cv2
import ctypes
from jigsaw.pieces_detection import extract_pieces
from jigsaw.solving import make_solution_array, solve_on_contours
import psutil


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")
        self.root.geometry("800x600")  # Set initial size

        self.image_button = tk.Button(self.root, text="Choose Image", command=self.choose_image)
        self.image_button.pack(padx=10, pady=(20, 10))  # Adjust pady for top padding

        self.img_path_label = tk.Label(self.root, text='Path: ')
        self.img_path_label.pack(padx=10, pady=(20, 10))  # Adjust pady for top padding

        self.bg_color_label = tk.Label(self.root, text='Background Color: ')
        self.bg_color_label.pack(padx=10, pady=(0, 10))  # Adjust pady for top padding

        # Add an attribute to track whether both conditions are met
        self.conditions_met = False

        # Create the processing button and set its state to DISABLED
        self.process_button = tk.Button(self.root, text="Process and display", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=(0, 10))
        self.pieces_len = tk.Label(self.root, text=' ')
        self.pieces_len.pack(padx=10, pady=(0, 10))  # Adjust pady for top padding
        # Add an attribute to store the selected color
        self.selected_color = None

    def choose_image(self):
        img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if img_path:
            import cv2
            print(cv2.__version__)
            self.img_path_label.config(text='Path: ' + img_path)
            from UI.color_selector import ColorSelector
            color_selector = ColorSelector(self.root, img_path)
            color_selector.show_color_selector()
            print("Color selector window closed")

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
        # Dynamically increase the memory limit (Windows-specific)
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        img_path = self.img_path_label.cget("text").split(": ")[1]
        print(img_path)
        img = cv2.imread(img_path)
        bgr_selected_color = self.selected_color
        # Get the current process ID
        pid = os.getpid()
        py = psutil.Process(pid)
        # Get memory usage
        memory_info = py.memory_info()
        print(f"Memory used: {memory_info.rss / 1024 / 1024:.2f} MB")
        left_up_piece, right_up_piece, left_down_piece, right_down_piece,center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces,w,h = extract_pieces(img, bgr_selected_color)
        total_pieces_count = 4 + len(center_up_pieces) + len(center_down_pieces) + len(center_left_pieces) + len(center_right_pieces) + len(center_pieces)
        self.pieces_len.config(text='Image processing complete. Pieces detected : ' + str(total_pieces_count))
        initial_w=240
        initial_h=240
        matched_pieces,grid_image = make_solution_array(center_left_pieces,center_up_pieces,initial_w=240,initial_h=240)
        solve_on_contours(matched_pieces,grid_image,initial_w,initial_h,left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h)
        print("Image processing complete.")

        
