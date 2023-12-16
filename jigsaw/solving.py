import numpy as np
import cv2
from pieces_detection import extract_pieces

def get_bump_direction_left_right(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    curvature = []
    for i in range(1, len(smoothed_contour) - 1):
        curvature.append(np.cross(smoothed_contour[i - 1] - smoothed_contour[i], smoothed_contour[i + 1] - smoothed_contour[i]))
    avg_curvature = np.mean(curvature)
    if avg_curvature > 0:
        return 1
    elif avg_curvature < 0:
        return 2
    else:
        return 3
    
def get_bump_direction_top_bottom(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    min_points_threshold = 3
    if len(smoothed_contour) < min_points_threshold:
        return 3
    centroid = np.mean(smoothed_contour[:, 0, :], axis=0)
    orientation = np.sign(centroid[1] - smoothed_contour[0, 0, 1])
    if orientation < 0:
        return 1
    elif  orientation > 0:
        return 2
    else:
        return 3


def solve_on_contours(left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h):
    solved_img = np.zeros((h, w, 3), dtype=np.uint8)
    ######################################################################
                                ##  TOP LINE ##
    ######################################################################
    #left_up
    left_up_piece_img = left_up_piece[0].sub_img
    solved_img[:left_up_piece_img.shape[0], :left_up_piece_img.shape[1]] = left_up_piece_img
    matched_pieces = [[]]
    matched_pieces[0] = [left_up_piece[0]]
    row_count = len(center_left_pieces) 
    start_col = 0
    start_row = 0
    #center_up
    while True :
        top_match = (float('inf'), None)
        for i, up_piece in enumerate(center_up_pieces):
            if get_bump_direction_left_right(matched_pieces[0][-1].right_contour)==get_bump_direction_left_right(up_piece.left_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[0][-1].right_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[0][-1].right_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(up_piece.left_contour, True)
                approx_compared_contour = cv2.approxPolyDP(up_piece.left_contour, epsilon, True)
                score = cv2.matchShapes(approx_left_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I2, 0.0)
                if score < top_match[0]:
                    top_match = (score, up_piece)
        if top_match[1]:        
            matched_piece_img = top_match[1].sub_img
            start_col = matched_pieces[0][-1].w +  start_col
            end_col = start_col + matched_piece_img.shape[1]
            solved_img[:matched_piece_img.shape[0], start_col:end_col] = matched_piece_img
        if len(center_up_pieces) == 1 :
            break
        matched_pieces[0].append(top_match[1])
        center_up_pieces.remove(top_match[1])
    #right_up
    right_up_piece_img = right_up_piece[0].sub_img
    start_col_right = matched_pieces[0][-1].w + start_col
    end_col_right = start_col_right + right_up_piece_img.shape[1]
    solved_img[:right_up_piece_img.shape[0], start_col_right:end_col_right] = right_up_piece_img
    matched_pieces[0].append(right_up_piece[0])
    ###########################################################################
                                ## LEFT LINE ##
    ###########################################################################
    #center_left
    counter = 0
    score = float('inf')
    while True :
        top_match = (float('inf'), None)
        for i, left_piece in enumerate(center_left_pieces):
            if get_bump_direction_top_bottom(matched_pieces[counter][0].bottom_contour)== get_bump_direction_top_bottom(left_piece.top_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[counter][0].bottom_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[counter][0].bottom_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(left_piece.top_contour, True)
                approx_compared_contour = cv2.approxPolyDP(left_piece.top_contour, epsilon, True)
                score = cv2.matchShapes(approx_left_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I2, 0.0)
            if score < top_match[0]:
                top_match = (score, left_piece)
        if top_match[1]:        
            matched_piece_img = top_match[1].sub_img
            start_row = matched_pieces[counter][0].h +  start_row
            end_row = start_row + matched_piece_img.shape[0]
            solved_img[start_row:end_row, :matched_piece_img.shape[1]] = matched_piece_img
        counter += 1
        matched_pieces.append([])
        matched_pieces[counter].append(top_match[1])
        if top_match[1] in center_left_pieces:
            center_left_pieces.remove(top_match[1])
        else:
            print(f"Warning: {top_match[1]} not found in center_left_pieces")
        #down_left
        if counter == row_count :
            matched_piece_img = left_down_piece[0].sub_img
            start_row = left_down_piece[0].h +  start_row
            end_row = start_row + matched_piece_img.shape[0]
            solved_img[start_row:end_row, :matched_piece_img.shape[1]] = matched_piece_img
            matched_pieces.append([])
            matched_pieces[counter+1].append(left_down_piece[0]) 
            break
    cv2.imshow('Solved Image', solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()