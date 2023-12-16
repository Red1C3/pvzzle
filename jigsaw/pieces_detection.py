import cv2
import copy
import numpy as np
from utils import img_utils
from jigsaw.pieces_types import PieceType
from jigsaw.piece import Piece

direction_mapping = {
    (0, 0, 0, 0): PieceType.CENTER,
    (1, 0, 0, 0): PieceType.CENTER_UP,
    (0, 1, 0, 0): PieceType.CENTER_DOWN,
    (0, 0, 1, 0): PieceType.CENTER_RIGHT,
    (0, 0, 0, 1): PieceType.CENTER_LEFT,
    (1, 0, 0, 1): PieceType.LEFT_UP,
    (1, 0, 1, 0): PieceType.RIGHT_UP,
    (0, 1, 1, 0): PieceType.RIGHT_DOWN,
    (0, 1, 0, 1): PieceType.LEFT_DOWN
}

                ###   SET PIECE TYPE  ###
        
def set_piece_type(piece):
    sub_img_bbox = piece.x, piece.y, piece.w, piece.h
    mask = piece.mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # direction_holder = [up, down, right, left]; this is the direction of the piece type
    direction_holder = [0, 0, 0, 0]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                if x2 < sub_img_bbox[0] + 10:
                    direction_holder[3] = 1
                elif x1 > sub_img_bbox[0] + sub_img_bbox[2] - 10:
                    direction_holder[2] = 1
            if y2 == y1:
                if y2 < sub_img_bbox[1] + 10:
                    direction_holder[0] = 1
                elif y1 > sub_img_bbox[1] + sub_img_bbox[3] - 10:
                    direction_holder[1] = 1
    piece.type = direction_mapping[tuple(direction_holder)]
    
            ###  DETECT CORNERS & SET CONTOUR ON EVERY DIRECTION BASED ON THE MAIN CONTOUR   ###
   
def contour_splitter(piece):
    corners_of_border = np.array([[0, 0], [0, piece.h], [piece.w, piece.h],[piece.w, 0] ])
    contour_points = piece.contour.reshape(-1, 2)
    len_contour_points=len(contour_points)
    closest_points = []
    #Indexs= []
    left_side=[]
    bottom_side=[]
    right_side=[]
    top_side=[]
    image = piece.mask[piece.y - 2:piece.y + piece.h + 2, piece.x -2 :piece.x + piece.w + 2]
    corners_detected = cv2.goodFeaturesToTrack(image, maxCorners=10, qualityLevel=0.01, minDistance=40)
    corners_detected = np.int0(corners_detected)
    closest_points = []
    for border_corner in corners_of_border:
        min_distance = float('inf')  
        closest_point = None
        for detected_point in corners_detected - np.array([2, 2]):
            distance = np.linalg.norm(border_corner - detected_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = detected_point
        closest_points.append(closest_point)
    closest_points = np.vstack(closest_points)
    Indexs = [np.argmin(np.linalg.norm(contour_points - closest_point, axis=1)) for closest_point in closest_points]
    left_side = contour_points [Indexs[0]:Indexs[1]+1]
    bottom_side = contour_points [Indexs[1]:Indexs[2]+1]
    right_side= contour_points [Indexs[2]:Indexs[3]+1]
    t1=contour_points [:Indexs[0]+1]
    t2=contour_points [Indexs[3]:len_contour_points+1]
    t1=t1[::-1]
    t2=t2[::-1]
    top_side = np.concatenate((t1, t2))
    piece.left_contour=left_side
    piece.bottom_contour=bottom_side
    piece.right_contour=right_side[::-1]
    piece.top_contour=top_side
    piece.corners=closest_points



            ### EXTRACT PIECES WITH ALL FEAUTERS SET ###


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
    img_contours = copy.deepcopy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 255), thickness=2)
    img_contours_resized = cv2.resize(img_contours, (int(img_contours.shape[1]*0.4), int(img_contours.shape[0] *0.4)))
    cv2.imshow('Contours of detected pieces', img_contours_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    masks = []
    for contour in contours:
        mask=np.zeros(np.shape(img_bin),'uint8')
        mask=cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
        masks.append(mask)

    min_piece_area = 5000
    left_up_piece = []
    right_up_piece = []
    left_down_piece = []
    right_down_piece = []
    center_up_pieces = []
    center_down_pieces = []
    center_left_pieces = []
    center_right_pieces = []
    center_pieces = []
    for mask, contour in zip(masks, contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_piece_area:
            mask = mask.astype(np.uint8)
            mask_slice = mask[y :y + h , x :x + w ]
            img_slice = img[y :y + h , x :x + w ].astype(np.uint8)
            sub_img = cv2.bitwise_and(img_slice, img_slice, mask=mask_slice)
            contour_normalized = contour - np.array([x, y])
            piece = Piece(x, y, w, h,sub_img,mask,contour_normalized,[],[],[],[],[],PieceType.UNKNOWN)
            set_piece_type(piece)
            contour_splitter(piece)
            if piece.type == PieceType.CENTER:
                center_pieces.append(piece)
            elif piece.type == PieceType.CENTER_UP:
                center_up_pieces.append(piece)
            elif piece.type == PieceType.CENTER_DOWN:
                center_down_pieces.append(piece)
            elif piece.type == PieceType.CENTER_LEFT:
                center_left_pieces.append(piece)
            elif piece.type == PieceType.CENTER_RIGHT:
                center_right_pieces.append(piece)
            elif piece.type == PieceType.LEFT_UP:
                left_up_piece.append(piece)
            elif piece.type == PieceType.RIGHT_UP:
                right_up_piece.append(piece)
            elif piece.type == PieceType.LEFT_DOWN:
                left_down_piece.append(piece)
            elif piece.type == PieceType.RIGHT_DOWN:
                right_down_piece.append(piece)
            
            mask = None
            piece = None
    contours = None
    return left_up_piece, right_up_piece, left_down_piece, right_down_piece,center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces,img.shape[1], img.shape[0]
