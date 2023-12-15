import numpy as np
import cv2
from pieces_detection import extract_pieces

def get_bump_direction(contour):
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
        return 0

def solve_on_contours(left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h):
    solved_img = np.zeros((h, w, 3), dtype=np.uint8)
    left_up_piece_img = left_up_piece[0].sub_img
    solved_img[:left_up_piece_img.shape[0], :left_up_piece_img.shape[1]] = left_up_piece_img
    matched_pieces = [left_up_piece[0]]  
    start_col=0
    while True :
        top_match = (float('inf'), None)
        for i, up_piece in enumerate(center_up_pieces):
            print(f"Comparing with center_up piece {i + 1}")
            if get_bump_direction(matched_pieces[-1].right_contour)==get_bump_direction(up_piece.left_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[-1].right_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[-1].right_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(up_piece.left_contour, True)
                approx_compared_contour = cv2.approxPolyDP(up_piece.left_contour, epsilon, True)
                score = cv2.matchShapes(approx_left_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I2, 0.0)
                print(f"Score {i + 1}: {score}")
                if score < top_match[0]:
                    top_match = (score, up_piece)
        if top_match[1]:        
            matched_piece_img = top_match[1].sub_img
            start_col = matched_pieces[-1].w +  start_col
            end_col = start_col + matched_piece_img.shape[1]
            solved_img[:matched_piece_img.shape[0], start_col:end_col] = matched_piece_img
        if len(center_up_pieces) == 1 :
            break
        matched_pieces.append(top_match[1])
        center_up_pieces.remove(top_match[1])
        right_up_piece_img = right_up_piece[0].sub_img
    start_col_right = matched_pieces[-1].w + start_col
    end_col_right = start_col_right + right_up_piece_img.shape[1]
    solved_img[:right_up_piece_img.shape[0], start_col_right:end_col_right] = right_up_piece_img
    matched_pieces.append(right_up_piece[0])
    cv2.imshow('Solved Image', solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
