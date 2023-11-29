import cv2
import numpy as np
from utils import img_utils

#NOTE Not finished yet
# Assumes background is of a black solid color not close to the pieces's color
def extract_pieces(img):
    img_blur=cv2.GaussianBlur(img,(5,5),None)
    img_gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    _,img_bin=cv2.threshold(img_gray,5,255,cv2.THRESH_BINARY)
    img_bin=cv2.GaussianBlur(img_bin,(5,5),None)
    contours,hiarachy=cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    masks=[]
    for contour in contours:
        mask=np.zeros(np.shape(img_bin),'uint8')
        mask=cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
        masks.append(mask)
    pieces=[]
    for mask,contour in zip(masks,contours):
        piece=cv2.bitwise_and(img,img,mask=mask)
        x,y,w,h=cv2.boundingRect(contour)
        piece=piece[y:y+h,x:x+w]
        pieces.append(piece)
    return pieces
