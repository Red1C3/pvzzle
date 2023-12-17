import cv2
from sklearn.cluster import KMeans

from jigsaw.pieces_detection import extract_pieces
from jigsaw.set_piece_type import set_piece_type


class Jigsaw:
    def __init__(self, colored_img, hint=None):
        self.pieces = extract_pieces(colored_img)
        for piece in self.pieces:
            set_piece_type(piece)
            piece.sift = piece.sift_features()
        if hint is not None:
            self.hint = hint
            sift = cv2.SIFT_create()
            self.hint_sift_features = sift.detectAndCompute(hint, None)

            for piece in self.pieces:
                kps1, des1 = piece.sift
                kps2, des2 = self.hint_sift_features

                matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                knn_matches = matcher.knnMatch(des1, des2, 2)
                ratio_threshold = 0.7
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                good_matches = sorted(good_matches, key=lambda x: x.distance)[:10]  # Take only 10 best features
                hint_match_points = []
                for match in good_matches:
                    hint_match_points.append((kps2[match.trainIdx].pt, match.distance))
                piece.hint_match_points = sorted(hint_match_points, key=lambda x: x[1])

    def cluster(self):
        match_points = []
        for piece in self.pieces:
            for point in piece.hint_match_points:
                match_points.append(point[0])
        k_means = KMeans(len(self.pieces), n_init=10)
        estimator = k_means.fit(match_points)
        for piece in self.pieces:
            matches_clusters = []
            for point in piece.hint_match_points:
                matches_clusters.append(estimator.predict([point[0]])[0])
            print(matches_clusters)
