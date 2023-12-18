from jigsaw.piece import Piece
import numpy as np
import cv2

def get_bump_direction_left_right(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    min_points_threshold = 3
    if len(smoothed_contour) < min_points_threshold:
        return 3
    centroid = np.mean(smoothed_contour[:, 0, :], axis=0)
    orientation = np.sign(centroid[0] - smoothed_contour[0, 0, 0])
    if orientation < 0:
        return 1
    elif orientation > 0:
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

def make_solution_array(center_left_pieces,center_up_pieces,initial_w=240,initial_h=240):
    row_count = len(center_left_pieces)+2
    column_count = len(center_up_pieces)+2
    initial_sub_img = np.zeros((initial_w, initial_h,3), dtype=np.uint8)
    initial_piece = Piece(x=0, y=0, w=initial_w, h=initial_h, sub_img=initial_sub_img, mask=None, contour=None)
    matched_pieces = [[initial_piece for _ in range(column_count)] for _ in range(row_count)]
    grid_size = (initial_h *row_count , initial_w * column_count)
    grid_image = np.zeros((grid_size[0], grid_size[1],3), dtype=np.uint8)
    return matched_pieces,grid_image

def display_answer(matched_pieces,row_count,column_count,grid_image,initial_w,initial_h):    
    for i in range(row_count):
        for j in range(column_count):
            piece = matched_pieces[i][j]
            sub_img = piece.sub_img
            canvas = np.zeros((initial_h, initial_w, 3), dtype=np.uint8)
            offset_row = (initial_h - sub_img.shape[0]) // 2
            offset_col = (initial_w - sub_img.shape[1]) // 2
            canvas[offset_row:offset_row+sub_img.shape[0], offset_col:offset_col+sub_img.shape[1]] = sub_img
            start_row, start_col = i * initial_h, j * initial_w
            end_row, end_col = start_row + initial_h, start_col + initial_w
            grid_image[start_row:end_row, start_col:end_col] = canvas
    resize_factor = 2
    grid_size = (initial_h *row_count , initial_w * column_count)
    grid_image_resized = cv2.resize(grid_image, (grid_size[1] // resize_factor, grid_size[0] // resize_factor))
    cv2.imshow("Solved Grid", grid_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def solve_on_contours(matched_pieces,grid_image,initial_w,initial_h,left_up_piece, right_up_piece, left_down_piece, right_down_piece, center_up_pieces, center_down_pieces, center_left_pieces, center_right_pieces, center_pieces, w, h):
   
    row_count = len(center_left_pieces)+2
    column_count = len(center_up_pieces)+2
    
    #### CORNERS SOLVE ####
    
    matched_pieces[0][0] = left_up_piece[0]
    matched_pieces[0][column_count-1] = right_up_piece[0]
    matched_pieces[row_count-1][0] = left_down_piece[0]
    matched_pieces[row_count-1][column_count-1] = right_down_piece[0]

    #### CENTER_UP ####
    
    count = 0
    while count < column_count - 2 :
        score = float('inf')
        top_match = (float('inf'), None)
        score_contour = float('inf')
        score_color = float('inf')
        for i, up_piece in enumerate(center_up_pieces):
            if get_bump_direction_left_right(matched_pieces[0][count].right_contour)==get_bump_direction_left_right(up_piece.left_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[0][count].right_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[0][count].right_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(up_piece.left_contour, True)
                approx_compared_contour = cv2.approxPolyDP(up_piece.left_contour, epsilon, True)
                score_contour = cv2.matchShapes(approx_left_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                vec_left = matched_pieces[0][count].get_quantization_vector()
                vec_right = up_piece.get_quantization_vector()
                score_color = np.linalg.norm(np.array(vec_right) - np.array(vec_left))
                if score_contour > 10 :
                    score_contour = 10
                if score_color > 10 : 
                    score_color = 10
                score = 0.55 * score_contour + 0.45 * score_color
            if score < top_match[0]:
                top_match = (score, up_piece)
        count += 1
        if top_match[1] :
            matched_pieces[0][count] = top_match[1]
        if top_match[1] in center_up_pieces:
            center_up_pieces.remove(top_match[1])
        else:
            print(f"Warning: {top_match[1]} not found in center_up_pieces")
            break
    
    #### CENTER_DOWN ####    
    
    count = 0
    while count < column_count-2 :
        score = 100
        top_match = (float('inf'), None)
        score_contour = 100
        score_color = 100
        for i, down_piece in enumerate(center_down_pieces):
            if get_bump_direction_left_right(matched_pieces[row_count-1][count].right_contour)==get_bump_direction_left_right(down_piece.left_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[row_count-1][count].right_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[row_count-1][count].right_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(down_piece.left_contour, True)
                approx_right_piece_contour = cv2.approxPolyDP(down_piece.left_contour, epsilon, True)
                score_contour = cv2.matchShapes(approx_left_piece_contour, approx_right_piece_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                vec_left = matched_pieces[row_count-1][count].get_quantization_vector()
                vec_right = down_piece.get_quantization_vector()
                score_color = np.linalg.norm(np.array(vec_right) - np.array(vec_left))
                if score_contour > 10 :
                    score_contour = 10
                if score_color > 10 : 
                    score_color = 10
                score = 0.55 * score_contour + 0.45 * score_color
            if score < top_match[0]:
                top_match = (score, down_piece)        
        count += 1
        if top_match[1]:
            matched_pieces[row_count-1][count] = top_match[1]
        if top_match[1] in center_down_pieces:
            center_down_pieces.remove(top_match[1])
        else:
            print(f"Warning: {top_match[1]} not found in center_down_pieces")
            break
    
    #### LEFT COLOMN ####

    count = 0
    while count < row_count-2 :
        score = 100
        top_match = (float('inf'), None)
        score_contour = 100
        score_color = 100
        for i, left_piece in enumerate(center_left_pieces):
            if get_bump_direction_top_bottom(matched_pieces[count][0].bottom_contour)== get_bump_direction_top_bottom(left_piece.top_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[count][0].bottom_contour, True)
                approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[count][0].bottom_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(left_piece.top_contour, True)
                approx_compared_contour = cv2.approxPolyDP(left_piece.top_contour, epsilon, True)
                score_contour = cv2.matchShapes(approx_left_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                vec_left = matched_pieces[count][0].get_quantization_vector()
                vec_right = left_piece.get_quantization_vector()
                score_color = np.linalg.norm(np.array(vec_right) - np.array(vec_left))
                if score_contour > 10 :
                    score_contour = 10
                if score_color > 10 : 
                    score_color = 10
                score = 0.55 * score_contour + 0.45 * score_color
            if score < top_match[0]:
                top_match = (score, left_piece)
        count += 1
        if top_match[1]:        
            matched_pieces[count][0] = top_match[1]
        if top_match[1] in center_left_pieces:
            center_left_pieces.remove(top_match[1])
        else:
            print(f"Warning: {top_match[1]} not found in center_left_pieces")
            break
    
    #### Right COLOMN ####

    count = 0
    while count < row_count-2 :
        score = 100
        top_match = (float('inf'), None)
        score_contour = 100
        score_color = 100
        for i, right_piece in enumerate(center_right_pieces):
            if get_bump_direction_top_bottom(matched_pieces[count][column_count-1].bottom_contour)== get_bump_direction_top_bottom(right_piece.top_contour):
                epsilon = 0.1 * cv2.arcLength(matched_pieces[count][column_count-1].bottom_contour, True)
                approx_right_piece_contour = cv2.approxPolyDP(matched_pieces[count][column_count-1].bottom_contour, epsilon, True)
                epsilon = 0.1 * cv2.arcLength(right_piece.top_contour, True)
                approx_compared_contour = cv2.approxPolyDP(right_piece.top_contour, epsilon, True)
                score_contour = cv2.matchShapes(approx_right_piece_contour, approx_compared_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                vec_left = matched_pieces[count][column_count-1].get_quantization_vector()
                vec_right = right_piece.get_quantization_vector()
                score_color = np.linalg.norm(np.array(vec_right) - np.array(vec_left))
                if score_contour > 10 :
                    score_contour = 10
                if score_color > 10 : 
                    score_color = 10
                score = 0.55 * score_contour + 0.45 * score_color
            if score < top_match[0]:
                top_match = (score, right_piece)
        count += 1
        if top_match[1]:
             matched_pieces[count][column_count-1] = top_match[1]
        if top_match[1] in center_right_pieces:
            center_right_pieces.remove(top_match[1])
        else:
            print('ERROR, NOT SOLVED in center_right_pieces!')
            break
        #else:
            #print(f"Warning: {top_match[1]} not found in center_right_pieces")
    
    ### CENTER LINES with left bottom compare ###
    #done = True
    center_pieces_copy = center_pieces.copy()
    main_counter = 1
    while main_counter < row_count-1 :
        inside_counter = 0
        while inside_counter < column_count-2 :  
            score = 100
            top_match = (float('inf'), None)
            score_contour = 100
            score_color = 100
            for i, center_piece in enumerate(center_pieces_copy):
                if get_bump_direction_left_right(matched_pieces[main_counter][inside_counter].right_contour)==get_bump_direction_left_right(center_piece.left_contour):
                    if get_bump_direction_top_bottom(matched_pieces[main_counter-1][inside_counter+1].bottom_contour)== get_bump_direction_top_bottom(center_piece.top_contour):
                        print(f'HERE {i}')
                        #score 1 for left contour
                        epsilon = 0.1 * cv2.arcLength(matched_pieces[main_counter][inside_counter].right_contour, True)
                        approx_left_piece_contour = cv2.approxPolyDP(matched_pieces[main_counter][inside_counter].right_contour, epsilon, True)
                        epsilon = 0.1 * cv2.arcLength(center_piece.left_contour, True)
                        approx_left_compared_contour = cv2.approxPolyDP(center_piece.left_contour, epsilon, True)
                        score1 = cv2.matchShapes(approx_left_piece_contour, approx_left_compared_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                        #score 2 for bottom contour
                        epsilon = 0.1 * cv2.arcLength(matched_pieces[main_counter-1][inside_counter+1].bottom_contour, True)
                        approx_bottom_piece_contour = cv2.approxPolyDP(matched_pieces[main_counter-1][inside_counter+1].bottom_contour, epsilon, True)
                        epsilon = 0.1 * cv2.arcLength(center_piece.top_contour, True)
                        approx_bottom_compared_contour = cv2.approxPolyDP(center_piece.top_contour, epsilon, True)
                        score2 = cv2.matchShapes(approx_bottom_piece_contour, approx_bottom_compared_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                        ##total contour score
                        if score1 > 10 :
                            score1 = 10
                        if score2 > 10 :
                            score2 = 10
                        score_contour = (score1 + score2)/2
                        vec_left = matched_pieces[main_counter][inside_counter].get_quantization_vector()
                        vec_right = center_piece.get_quantization_vector()
                        score_color1 = np.linalg.norm(np.array(vec_left) - np.array(vec_right))
                        vec_up = matched_pieces[main_counter-1][inside_counter+1].get_quantization_vector()
                        vec_down = center_piece.get_quantization_vector()
                        score_color2 = np.linalg.norm(np.array(vec_up) - np.array(vec_down))
                        score_color = (score_color1 + score_color2)/2
                        score = 0.55 * score_contour + 0.45 * score_color
                    if score < top_match[0]:
                        top_match = (score, center_piece)
            inside_counter +=1
            if top_match[1]:    
                matched_pieces[main_counter][inside_counter]=top_match[1]
            if top_match[1] in center_pieces:
                center_pieces_copy.remove(top_match[1])
            else:
                print('ERROR, NOT SOLVED!')
                main_counter += 1
                break
        main_counter += 1
    display_answer(matched_pieces,row_count,column_count,grid_image,initial_w,initial_h)
