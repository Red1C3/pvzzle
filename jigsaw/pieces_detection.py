import cv2
import numpy as np
from utils import img_utils
from jigsaw.pieces_types import PieceType
from jigsaw.piece import Piece

# Assumes background is of a black solid color not close to the pieces's color
def extract_pieces(img,background_color):
    bgr_background_color = (background_color[2], background_color[1], background_color[0])
    background_mask = cv2.inRange(img, bgr_background_color, bgr_background_color)
    foreground_mask = cv2.bitwise_not(background_mask)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_subtracted = cv2.bitwise_and(img_gray, img_gray, mask=foreground_mask)
    _, img_bin = cv2.threshold(img_subtracted, 0, 255, cv2.THRESH_BINARY)
    kernel_closing = np.ones((9, 9), np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_closing)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    masks = []
    for contour in contours:
        mask=np.zeros(np.shape(img_bin),'uint8')
        mask=cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
        masks.append(mask)
    min_piece_area = 5000
    pieces = []
    for mask, contour in zip(masks, contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_piece_area:
            mask = mask.astype(np.uint8)
            mask_slice = mask[y:y + w, x:x + h]
            img_slice = img[y:y + w, x:x + h].astype(np.uint8)
            piece = cv2.bitwise_and(img_slice, img_slice, mask=mask_slice)
            contour_normalized = contour - np.array([x, y])
            pieces.append(Piece(x, y, w, h, piece, mask, contour_normalized, PieceType.UNKNOWN)) 
            mask = None
            piece = None
    contours = None
    return pieces

