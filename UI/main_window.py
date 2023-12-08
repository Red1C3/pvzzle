import tkinter as tk
from tkinter import filedialog



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

    def choose_image(self):
        img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if img_path:
            self.img_path_label.config(text='Path: ' + img_path)
            from UI.color_selector import ColorSelector
            color_selector = ColorSelector(self.root, img_path)
            color_selector.show_color_selector()
            print("Color selector window closed")
            # Access the selected color after the window is closed
            if color_selector.color_selected:
                selected_color = color_selector.color_selected
                selected_color_str = str(selected_color)
                self.bg_color_label.config(text= 'Background Color: '+ selected_color_str)
            else:
                print("Color selection canceled.")

if __name__ == "__main__":
    root = tk.Tk()
    main_window = MainWindow(root)
    root.mainloop()