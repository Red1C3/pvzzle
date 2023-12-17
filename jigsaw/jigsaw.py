import cv2
import numpy as np
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

                matcher = cv2.DescriptorMatcher_create(
                    cv2.DescriptorMatcher_FLANNBASED)
                knn_matches = matcher.knnMatch(des1, des2, 2)
                ratio_threshold = 0.7
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                good_matches = sorted(good_matches, key=lambda x: x.distance)[
                    :10]  # Take only 10 best features
                hint_match_points = []
                for match in good_matches:
                    hint_match_points.append(
                        (kps2[match.trainIdx].pt, match.distance))
                piece.hint_match_points = sorted(
                    hint_match_points, key=lambda x: x[1])

    def cluster(self):
        match_points = []
        for piece in self.pieces:
            for point in piece.hint_match_points:
                match_points.append(point[0])
        k_means = KMeans(len(self.pieces), n_init=10)
        estimator = k_means.fit(match_points)
        clusters = {i: [None, 0] for i in range(len(self.pieces))}
        for piece in self.pieces:
            matches_clusters = []
            for point in piece.hint_match_points:
                matches_clusters.append(estimator.predict([point[0]])[0])
            unique, counts = np.unique(matches_clusters, return_counts=True)
            unique_counts = dict(zip(unique, counts))
            for k, v in clusters.items():
                if k not in unique_counts.keys():
                    continue
                if v[0] is None:
                    v[0] = piece
                    v[1] = unique_counts[k]
                elif unique_counts[k] > v[1]:
                    v[0] = piece
                    v[1] = unique_counts[k]
        return clusters.values(), estimator.cluster_centers_

    def clusters_img(self, offset=(100, 100)):
        clusters = self.cluster()
        cimg = np.zeros((self.hint.shape[0]*2, self.hint.shape[1]*2, 3),'uint8')
        for piece, center in zip(clusters[0], clusters[1]):
            piece = piece[0]
            pimg = piece.sub_img
            pshape = pimg.shape
            cimg[round(center[1]-pshape[0]/2)+offset[0]:round(center[1]+pshape[0]/2)+offset[0],
                 round(center[0]-pshape[1]/2)+offset[1]:round(center[0]+pshape[1]/2)+offset[1]] += pimg
        return cimg
