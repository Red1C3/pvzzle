# Sizes are defined (width,height)
class Grid:
    def __init__(self, img_colored, size):
        self.img = img_colored
        self.size = size
        self.piece_size = ((img_colored.shape[1]) // size[0], (img_colored.shape[0]) // size[1])

    def get_piece(self, coordinates):
        c = coordinates
        return self.img[c[1] * self.piece_size[1]:c[1] * self.piece_size[1] + self.piece_size[1],
               c[0] * self.piece_size[0]:c[0] * self.piece_size[0] + self.piece_size[0]]
