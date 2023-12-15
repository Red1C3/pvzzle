import cv2
class Piece:
    def __init__(self,img):
        self.img=img
        self.up_dict={}
        self.left_dict={}

    def get_sift_features(self, features_count=20):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=features_count)
        return sift.detectAndCompute(gray,None)
