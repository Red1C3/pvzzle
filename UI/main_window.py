import tkinter as tk
from tkinter import filedialog
 

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")
        self.root.geometry("800x600")  # Set initial size

        self.image_button = tk.Button(self.root, text="Choose Image", command=self.choose_image)
        self.image_button.pack()
    def choose_image(self):
        img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if img_path:
            from UI.color_selector import ColorSelector
            color_selector = ColorSelector(self.root, img_path)
            color_selector.show_color_selector()

            # Access the selected color after the window is closed
            if color_selector.color_selected:
                selected_color = color_selector.color_selected
                print("Selected Color:", selected_color)
            else:
                print("Color selection canceled.")
            

