import cv2
import numpy as np
from utils import img_utils
from jigsaw.pieces_types import PieceType
from jigsaw.piece import Piece

# Assumes background is of a black solid color not close to the pieces's color
def extract_pieces(img):
    img_blur=cv2.GaussianBlur(img,(5,5),None)
    img_gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    _,img_bin=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)
    kernel_closing = np.ones((9, 9), np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_closing)
    img_bin=cv2.GaussianBlur(img_bin,(5,5),None)
    contours,hiarachy=cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    masks=[]
    for contour in contours:
        mask=np.zeros(np.shape(img_bin),'uint8')
        mask=cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
        masks.append(mask)
    min_piece_area = 5000 
    pieces = []
    for mask, contour in zip(masks, contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_piece_area:
            piece = cv2.bitwise_and(img, img, mask=mask)
            piece = piece[y:y + h, x:x + w]
            contour_normalized = contour - np.array([x, y])
            pieces.append(Piece(x, y, w, h, piece, mask, contour_normalized, PieceType.UNKNOWN))  
    return pieces
