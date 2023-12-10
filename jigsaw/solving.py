import numpy as np
import cv2

def solve_on_contours(left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h):
    solved_img = np.zeros((h, w, 3), dtype=np.uint8)
    left_up_piece_img = left_up_piece[0].sub_img
    solved_img[:left_up_piece_img.shape[0], :left_up_piece_img.shape[1]] = left_up_piece_img

    # Initialize variables to store top matches
    top_matches = [(float('inf'), None)] * 5
    # Compare left_up_piece[0].right_contour with the left contours of center_up_pieces
    for up_piece in center_up_pieces:
       # normalized_left_up_contour = normalize_contour(left_up_piece[0].right_contour)
        #print(normalized_left_up_contour)
        #normalized_up_piece_contour = normalize_contour(up_piece.left_contour)
        #print(normalized_up_piece_contour)
        score = cv2.matchShapes(left_up_piece[0].right_contour, up_piece.left_contour, cv2.CONTOURS_MATCH_I1, 0.1)
        print(score)
        # Update top_matches if the current score is better
        for i, (top_score, top_piece) in enumerate(top_matches):
            if score < top_score:
                top_matches.insert(i, (score, up_piece))
                top_matches.pop()
                break
    print(top_matches)
    # Display the top 3 matches to the right of left_up_piece
    for i, (score, match_piece) in enumerate(top_matches):
        if match_piece:
            left_up_piece_width = left_up_piece_img.shape[1]
            solved_img[i * match_piece.sub_img.shape[0]:(i + 1) * match_piece.sub_img.shape[0], 
                       left_up_piece_width:left_up_piece_width + match_piece.sub_img.shape[1]] = match_piece.sub_img

    # Show the resulting image
    cv2.imshow('Solved Image', solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# solve_on_contours(left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h)