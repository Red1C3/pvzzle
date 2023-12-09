import cv2
import numpy as np
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
    
def contour_splitter(piece):
    corners = np.array([[0, 0], [0, piece.h], [piece.w, piece.h],[piece.w, 0] ])
    contour_points = piece.contour.reshape(-1, 2)
    len_contour_points=len(contour_points)
    closest_points = []
    Indexs= []
    left_side=[]
    bottom_side=[]
    right_side=[]
    top_side=[]
    for corner in corners:
        # Calculate distances between the corner and all contour points
        distances = np.linalg.norm(contour_points - corner, axis=1)
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        # Get the closest point
        closest_point = tuple(contour_points[closest_index])
        closest_points.append(closest_point)
        Indexs.append(closest_index)
    num_of_left=Indexs[1]-Indexs[0]
    num_of_bottom=Indexs[2]-Indexs[1]
    num_of_right=Indexs[3]-Indexs[2]
    num_of_top=len_contour_points-Indexs[3]
    print(num_of_left,num_of_bottom,num_of_right,num_of_top)
    # handle every status for edges
    left_side = contour_points [Indexs[0]:Indexs[1]+1]
    bottom_side = contour_points [Indexs[1]:Indexs[2]+1]
    right_side= contour_points [Indexs[2]:Indexs[3]+1]
    t1=contour_points [:Indexs[0]+1]
    t2=contour_points [Indexs[3]:len_contour_points+1]
    top_side.extend(t1)
    top_side.extend(t2) 
    # Assignment of edge arrays
    piece.left_contour=left_side
    piece.bottom_contour=bottom_side
    piece.right_contour=right_side
    piece.top_contour=top_side
    piece.corners=closest_points
    
    '''
                 ##  VISUALISATION  ##
    
    # Draw the points on the piece image
    piece_img_with_points = piece.sub_img.copy()
    #  1     points: left_up -> left_down 
    for point in left_side:
        cv2.circle(piece_img_with_points, point, 5, (0, 100, 100), -1)  
    #   2    points: left_down -> right_down 
    for point in bottom_side:
        cv2.circle(piece_img_with_points, point, 5, (100, 100, 0), -1)  
    #   3    points: right_down -> right_up
    for point in right_side:
        cv2.circle(piece_img_with_points, point, 5, (100, 0, 100), -1)  
    #   4    points: right_up -> left_up  
    for point in top_side:
        cv2.circle(piece_img_with_points, point, 5, (100, 100, 100), -1)  
    for point in closest_points:
        cv2.circle(piece_img_with_points, point, 5, (0, 0, 255), -1)   
    # Display the piece image with points
    cv2.imshow('Piece Image with Points', piece_img_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
          

    '''
    #  THE RETURN STATMENT NOT NEEDED #
    # return closest_points,Indexs
