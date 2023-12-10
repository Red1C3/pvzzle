import cv2
import numpy as np

class CountGridPieces:
    def __init__(self, image_path, canny_threshold1=80, canny_threshold2=200, hough_threshold=200):
        self.image_path = image_path
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_threshold = hough_threshold

    def read_image(self):
        self.image = cv2.imread(self.image_path)

    def preprocess_image(self):
        blurred = cv2.GaussianBlur(self.image, (3, 3), 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        self.lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=self.hough_threshold)

    def filter_lines(self):
        self.filtered_lines = []
        for i in range(len(self.lines)):
            rho, theta = self.lines[i][0]

            if (np.pi / 4 < theta < 3 * np.pi / 4) or (0 <= theta < np.pi / 4) or (3 * np.pi / 4 <= theta <= np.pi):
                if 10 < rho < min(self.image.shape[0], self.image.shape[1]) - 10:
                    add_line = True
                    for filtered_line in self.filtered_lines:
                        rho_f, _ = filtered_line[0]
                        
                    if add_line:
                        self.filtered_lines.append(self.lines[i])

    def separate_lines(self):
        self.horizontal_lines = []
        self.vertical_lines = []

        self.vertical_lines.append(self.image.shape[1])
        self.horizontal_lines.append(self.image.shape[0])

        for line in self.filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if np.pi / 4 < theta < 3 * np.pi / 4:
                self.horizontal_lines.append(y0)
            else:
                self.vertical_lines.append(x0)

        self.horizontal_lines.append(0.0)
        self.vertical_lines.append(0.0)

        self.horizontal_lines.sort()
        self.vertical_lines.sort()

    def calculate_distances(self):
        self.horizontal_distances = [self.horizontal_lines[i + 1] - self.horizontal_lines[i] for i in
                                     range(len(self.horizontal_lines) - 1)]
        self.vertical_distances = [self.vertical_lines[i + 1] - self.vertical_lines[i] for i in
                                   range(len(self.vertical_lines) - 1)]

        self.horizontal_distances_rounded = [round(num, -1) for num in self.horizontal_distances]
        self.vertical_distances_rounded = [round(num, -1) for num in self.vertical_distances]

    def count_pieces(self):
        sum_of_elements_horizontal = sum(self.horizontal_distances_rounded)
        min_number_horizontal = min(self.horizontal_distances_rounded)
        self.horizontal_result = sum_of_elements_horizontal / min_number_horizontal

        sum_of_elements_vertical = sum(self.vertical_distances_rounded)
        min_number_vertical = min(self.vertical_distances_rounded)
        self.vertical_result = sum_of_elements_vertical / min_number_vertical

    def display_result(self):
        print("Number of pieces in every column: ", round(self.horizontal_result))
        print("Number of pieces in every row: ", round(self.vertical_result))

    def draw_lines(self):
        for line in self.filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def show_image(self):
        cv2.imshow('Detected Lines', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example Usage:
if __name__ == "__main__":
    image_path = 'samples/shuffled-8.PNG'
    counter = CountGridPieces(image_path)
    counter.read_image()
    counter.preprocess_image()
    counter.filter_lines()
    counter.separate_lines()
    counter.calculate_distances()
    counter.count_pieces()
    counter.display_result()
    counter.draw_lines()
    counter.show_image()
