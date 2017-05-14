import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from utils import *
import math


'''
MATCH_KEYPOINTS: Given two sets of descriptors corresponding to SIFT keypoints, 
find pairs of matching keypoints.

Note: Read Lowe's Keypoint matching, finding the closest keypoint is not
sufficient to find a match. thresh is the threshold for a valid match.

Arguments:
    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    threshold - The threshold which to accept from Lowe's Keypoint Matching
        algorithm

Returns:
    matches - An int ndarray of size (N, 2) of indices that for keypoints in 
        descriptors1 match which keypoints in descriptors2. For example, [7 5]
        would mean that the keypoint at index 7 of descriptors1 matches the
        keypoint at index 5 of descriptors2. Not every keypoint will necessarily
        have a match, so N is not the same as the number of rows in descriptors1
        or descriptors2. 
'''
def match_keypoints(descriptors1, descriptors2, threshold = 0.7):
    # TODO: Implement this method!
    matches_list = []
    # for each row of descriptors1, find Euclidean distances to each row of descriptor2
    for i in xrange(descriptors1.shape[0]):
        descpt1 = descriptors1[i]
        euc_dist = np.sqrt(np.sum((descriptors2 - descpt1)**2, axis=1))
        index_sort = np.argsort(euc_dist, axis=None)

        closest = index_sort[0]
        second_closest = index_sort[1]
        if (euc_dist[closest] < threshold * euc_dist[second_closest]):
            matches_list.append(i)
            matches_list.append(closest)

    matches = np.array(matches_list).reshape(-1, 2)
    return matches


'''
REFINE_MATCH: Filter out spurious matches between two images by using RANSAC
to find a projection matrix. 

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2. 

    reprojection_threshold - If the reprojection error is below this threshold,
        then we will count it as an inlier during the RANSAC process.

    num_iterations - The number of iterations we will run RANSAC for.

Returns:
    inliers - A vector of integer indices that correspond to the inliers of the
        final model found by RANSAC.

    model - The projection matrix H found by RANSAC that has the most number of
        inliers.
'''
def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
        num_iterations = 1000):
    # TODO: Implement this method!
    # best parameters to return
    best_model = None
    best_inliers = []
    best_count = 0

    # every sample provides two constraints and
    # H is a 3x3 matrix known up to scale
    sample_size = 4
    P = np.zeros((2 * sample_size, 9)) # refer to Note 1
    for i in xrange(num_iterations):
        sample_indexes = random.sample(range(0, matches.shape[0]), sample_size)
        sample = matches[sample_indexes, :] # (4x2)

        for index, elem in enumerate(sample):
            # two keypoints indexes
            point1_index = elem[0]
            point2_index = elem[1]

            # xi' = H(xi), xi: keypoints1(ui, vi), xi': keypoints2(ui', vi')
            point1 = keypoints1[point1_index, 0:2] # (ui, vi)
            point1 = np.append(point1, 1)          # (ui, vi, 1)
            point2 = keypoints2[point2_index, 0:2] # (ui', vi')
            ui_prime = point2[0]
            vi_prime = point2[1]

            # construct P matrix
            P[2*index, :] = np.reshape(np.array([point1, np.zeros(3), -ui_prime*point1]), -1)
            P[2*index+1, :] = np.reshape(np.array([np.zeros(3), point1, -vi_prime*point1]), -1)
        # solve
        U, s,VT = np.linalg.svd(P)
        H = VT[-1, :].reshape(3, 3)
        H /= H[2, 2]

        # evaluate H
        inliers = []
        count = 0
        for index, match in enumerate(matches):
            point1 = keypoints1[match[0], 0:2] # (ui, vi)
            point1 = np.append(point1, 1) # (ui, vi, 1)
            # project (ui, vi, 1) and get (ui', vi')
            # (ui, vi, 1) -H-> (u, v, w) -norm-> (ui', vi', 1) -drop-> (ui, vi)
            point2_pred = H.dot(point1)
            point2_pred /= point2_pred[2]
            point2_pred = point2_pred[0:2]
            # compare prediction and ground truth
            point2 = keypoints2[match[1], 0:2]
            err = np.sqrt(np.sum(np.square(point2 - point2_pred)))
            if err < reprojection_threshold:
                count += 1
                inliers.append(index)

        # update the best model
        if count > best_count:
            best_model = H
            best_inliers = inliers
            best_count = count

    return best_inliers, best_model

'''
GET_OBJECT_REGION: Get the parameters for each of the predicted object
bounding box in the image

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2.

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

    thresh - The threshold we use in Hough voting to state that we have found
        a valid object region.

Returns:
    cx - A list of the x location of the center of the bounding boxes

    cy - A list of the y location of the center of the bounding boxes

    w - A list of the width of the bounding boxes

    h - A list of the height of the bounding boxes

    orient - A list f the orientation of the bounding box. Note that the 
        theta provided by the SIFT keypoint is inverted. You will need to
        re-invert it.
'''
def get_object_region(keypoints1, keypoints2, matches, obj_bbox, thresh = 5, 
        nbins = 4):
    # TODO: Implement this method!
    raise Exception('Not Implemented Error')

'''
MATCH_OBJECT: The pipeline for matching an object in one image with another

Arguments:
    im1 - The first image read in as a ndarray of size (H, W, C).

    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    im2 - The second image read in as a ndarray of size (H, W, C).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def match_object(im1, descriptors1, keypoints1, im2, descriptors2, keypoints2,
        obj_bbox):
    # Part A
    descriptors1, keypoints1, = select_keypoints_in_bbox(descriptors1,
        keypoints1, obj_bbox)
    matches = match_keypoints(descriptors1, descriptors2)
    #plot_matches(im1, im2, keypoints1, keypoints2, matches)
    
    # Part B
    inliers, model = refine_match(keypoints1, keypoints2, matches)
    plot_matches(im1, im2, keypoints1, keypoints2, matches[inliers,:])

    # Part C
    cx, cy, w, h, orient = get_object_region(keypoints1, keypoints2,
        matches[inliers,:], obj_bbox)

    plot_bbox(cx, cy, w, h, orient, im2)

if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('SIFT_data.mat')
    images = data['stopim'][0]
    obj_bbox = data['obj_bbox'][0]
    keypoints = data['keypt'][0]
    descriptors = data['sift_desc'][0]
    
    np.random.seed(0)

    for i in [2, 1, 3, 4]:
        match_object(images[0], descriptors[0], keypoints[0], images[i],
            descriptors[i], keypoints[i], obj_bbox)
