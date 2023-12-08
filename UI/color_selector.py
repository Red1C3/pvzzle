import tkinter as tk
from PIL import Image, ImageTk
from UI.main_window import MainWindow 

class ColorSelector:
    def __init__(self, root, img_path):
        self.root = root
        self.img_path = img_path
        self.color_selected = None
        self.scaled_img = self.scale_image_for_display()
    
    def on_ok_button_click(self):
        self.color_selector_window.destroy()
    
    def show_color_selector(self):
        self.color_selector_window = tk.Toplevel(self.root)
        self.color_selector_window.title("Color Selector")
        tk_img = ImageTk.PhotoImage(self.scaled_img)
        self.image_label = tk.Label(self.color_selector_window, image=tk_img)
        self.image_label.image = tk_img
        self.image_label.pack()
        self.create_widgets(self.color_selector_window)
        self.color_selector_window.mainloop()

    def scale_image_for_display(self):
        max_width = 1366
        max_height = 760
        img = Image.open(self.img_path)
        width, height = img.size
        # Scale down if either width or height exceeds the maximum
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        return img

    def create_widgets(self, window):
        self.color_label = tk.Label(window, text="RGB: ")
        self.color_label.pack()
        ok_button = tk.Button(window, text="OK", command=self.on_ok_button_click)
        ok_button.pack()
        self.image_label.bind("<Button-1>", self.pick_color)

    def pick_color(self, event):
        x, y = event.x, event.y
        # Calculate the position on the original-sized image
        original_x = int(x / self.image_label.winfo_width() * self.scaled_img.width)
        original_y = int(y / self.image_label.winfo_height() * self.scaled_img.height)
        # Scale down the original image
        img = self.scaled_img
        # Calculate the position on the scaled image
        scaled_x = int(original_x / img.width * self.scaled_img.width)
        scaled_y = int(original_y / img.height * self.scaled_img.height)
        pixel_color = img.getpixel((scaled_x, scaled_y))
        self.color_selected = pixel_color
        self.color_label.config(text=f"RGB: {pixel_color}")


if __name__ == "__main__":
    root = tk.Tk()
    main_window = MainWindow(root)
    root.mainloop()