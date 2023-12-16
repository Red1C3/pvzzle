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
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.round(gray * (levels / 255)) * (255 / levels)
        return gray, levels

    def get_quantization_vector(self, levels=8):
        qimg, _ = self.get_quantized_img(levels)
        steps = np.linspace(0, 255, levels + 1)
        vec = []
        unique, counts = np.unique(qimg, return_counts=True)
        unique_counts = dict(zip(unique, counts))
        total_elements_count = qimg.shape[0] * qimg.shape[1]
        for step in steps:
            if step in unique_counts.keys():
                vec.append(unique_counts[step] / total_elements_count)
            else:
                vec.append(0)
        return vec
