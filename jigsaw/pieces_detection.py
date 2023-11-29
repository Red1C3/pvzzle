import cv2
from utils import img_utils

#NOTE Not finished yet
# Assumes background is of a black solid color not close to the pieces's color
def extract_pieces(img):
    img=cv2.GaussianBlur(img,(5,5),None)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img=cv2.threshold(img,5,255,cv2.THRESH_BINARY)
    img=cv2.GaussianBlur(img,(5,5),None)
    contours,hiarachy=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
