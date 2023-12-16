import cv2
class Piece:
    def __init__(self,img):
        self.img=img
        self.up_dict={}
        self.left_dict={}

    def get_sift_features(self):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(gray,None)
