import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    points_num = points2.shape[0]

    W = np.zeros((points_num, 9))
    for i in xrange(points_num):
        u1 = points1[i][0]
        v1 = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]
        W[i] = np.array([u1*u2, u2*v1, u2, v2*u1, v1*v2, v2, u1, v1, 1])

    # compute F_hat
    U, s, VT = np.linalg.svd(W, full_matrices=True)
    f = VT[-1, :]
    F_hat = np.reshape(f, (3, 3))

    # compute F
    U, s_hat, VT = np.linalg.svd(F_hat, full_matrices=True)
    s = np.zeros((3, 3))
    s[0][0] = s_hat[0]
    s[1][1] = s_hat[1]
    F = np.dot(U, np.dot(s, VT))

    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    N = points1.shape[0]
    points1_uv = points1[:, 0:2]
    points2_uv = points2[:, 0:2]

    # normalization
    mean1 = np.mean(points1_uv, axis=0)
    mean2 = np.mean(points2_uv, axis=0)

    points1_center = points1_uv - mean1
    points2_center = points2_uv - mean2

    scale1 = np.sqrt(2/(np.sum(points1_center**2)/N * 1.0))
    scale2 = np.sqrt(2/(np.sum(points2_center**2)/N * 1.0))

    T1 = np.array([[scale1, 0, -mean1[0] * scale1],
                   [0, scale1, -mean1[1] * scale1],
                   [0, 0, 1]])

    T2 = np.array([[scale2, 0, -mean2[0] * scale2],
                   [0, scale2, -mean2[0] * scale2],
                   [0, 0, 1]])

    q1 = T1.dot(points1.T).T; q2 = T2.dot(points2.T).T
    Fq = lls_eight_point_alg(q1, q2)
    F = T2.T.dot(Fq).dot(T1)

    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    plt.subplot(1,2,1)
    ln1 = F.T.dot(points2.T)
    for i in xrange(ln1.shape[1]):
        plt.plot([0, im1.shape[1]], [-ln1[2][i]*1.0/ln1[1][i], -(ln1[2][i]+ln1[0][i]*im1.shape[1])*1.0/ln1[1][i]], 'r')
        plt.plot([points1[i][0]], [points1[i][1]], 'b*')
    plt.imshow(im1, cmap='gray')

    plt.subplot(1,2,2)
    ln2 = F.dot(points1.T)
    for i in xrange(ln2.shape[1]):
        plt.plot([0, im2.shape[1]], [-ln2[2][i]*1.0/ln2[1][i], -(ln2[2][i]+ln2[0][i]*im2.shape[1])/ln2[1][i]], 'r')
        plt.plot([points2[i][0]], [points2[i][1]], 'b*')
    plt.imshow(im2, cmap='gray')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    l = F.T.dot(points2.T)
    dist_sum = 0.0
    points_num = points1.shape[0]

    for i in xrange(points_num):
        dist_sum += np.abs(points1[i][0]*l[0][i] + points1[i][1]*l[1][i] + l[2][i]) * 1.0 \
                    / np.sqrt(l[0][i]**2 + l[1][i]**2)
    return dist_sum / points_num

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plt.figure("Without Normalization")
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plt.figure("Normalized")
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
