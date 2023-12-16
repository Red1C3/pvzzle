import cv2
import numpy as np


class Piece:
    def __init__(self,img):
        self.img=img
        self.up_dict={}
        self.left_dict={}

    def get_sift_features(self):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(gray,None)

    def get_quantized_img(self, levels=8):
        quantized = np.round(self.img * (levels / 255)) * (255 / levels)
        return quantized, levels

    def get_quantization_vector(self, levels=8):
        qimg, _ = self.get_quantized_img(levels)
        steps = np.linspace(0, 255, levels + 1)
        vec = []

        unique, counts = np.unique(qimg[:, :, 0], return_counts=True)
        unique_counts_b = dict(zip(unique, counts))
        unique, counts = np.unique(qimg[:, :, 1], return_counts=True)
        unique_counts_g = dict(zip(unique, counts))
        unique, counts = np.unique(qimg[:, :, 2], return_counts=True)
        unique_counts_r = dict(zip(unique, counts))

        total_elements_count = qimg.shape[0] * qimg.shape[1]

        for step in steps:
            if step in unique_counts_b.keys():
                vec.append(unique_counts_b[step] / total_elements_count)
            else:
                vec.append(0)

        for step in steps:
            if step in unique_counts_g.keys():
                vec.append(unique_counts_g[step] / total_elements_count)
            else:
                vec.append(0)

        for step in steps:
            if step in unique_counts_r.keys():
                vec.append(unique_counts_r[step] / total_elements_count)
            else:
                vec.append(0)
        return vec
