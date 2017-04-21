"""CS231A Homework 1, Problem 2.

DATA FORMAT
In this problem, we provide and load the data for you. Recall that in the
original problem statement, there exists a grid of black squares on a white
background. We know how these black squares are setup, and thus can determine
the locations of specific points on the grid (namely the corners). We also have
images taken of the grid at a front image (where Z = 0) and a back image (where
Z = 150). The data we load for you consists of three parts: real_XY,
front_image, and back_image. For a corner (0,0), we may see it at the (137, 44)
pixel in the front image and the (148, 22) pixel in the back image. Thus, one
row of real_XY will contain the numpy array [0, 0], corresponding to the real
XY location (0, 0). The matching row in front_image will contain [137, 44] and
the matching row in back_image will contain [148, 22].
"""

import numpy as np

def compute_camera_matrix(real_XY, front_image, back_image):
    """Computes camera matrix given image and real-world coordinates.

    Args:
        real_XY: Each row corresponds to an actual point on the 2D plane.
        front_image: Each row is the pixel location in the front image (Z=0).
        back_image: Each row is the pixel location in the back image (Z=150).
    Returns:
        camera_matrix: The calibrated camera matrix (3x4 matrix).
    """
    img_num1 = front_image.shape[0]
    img_num2 = back_image.shape[0]

    x = np.zeros((2, img_num1+img_num2))
    for i in xrange(img_num1):
        x[:, i] = front_image[i, :].T
    for j in xrange(img_num2):
        x[:, j + img_num1] = back_image[j, :].T
    x_ones = np.ones((1, x.shape[1]))
    x = np.vstack((x, x_ones))

    X = np.zeros((2, img_num1+img_num2))
    for i in xrange(img_num1):
        X[:, i] = real_XY[i, :].T
    for j in xrange(img_num2):
        X[:, j + img_num1] = real_XY[j, :].T
    Z = np.zeros((1, X.shape[1]))
    for k in xrange(Z.shape[1]):
        if k >= img_num1:
            Z[:, k] = 150
    X = np.vstack((X, Z))
    X_ones = np.ones((1, X.shape[1]))
    X = np.vstack((X, X_ones))

    A = np.zeros((2 * (img_num1+img_num2), 8))
    for i in range(0, A.shape[0], 2):
        A[i, :] = np.hstack((X[:, i/2].T, [0, 0, 0, 0]))
        A[i + 1, :] = np.hstack(([0, 0, 0, 0], X[:, i/2].T))

    b = front_image[0].T
    for i in range(1, img_num1, 1):
        b = np.hstack((b, front_image[i].T))
    for j in range(img_num2):
        b = np.hstack((b, back_image[j].T))
    b = np.reshape(b, (2 * (img_num1 + img_num2), 1))

    p = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    p = np.reshape(p, (2, -1))
    camera_matrix = np.vstack((p, [0, 0, 0, 1]))

    return camera_matrix


def rms_error(camera_matrix, real_XY, front_image, back_image):
    """Computes RMS error of points reprojected into the images.

    Args:
        camera_matrix: The camera matrix of the calibrated camera.
        real_XY: Each row corresponds to an actual point on the 2D plane.
        front_image: Each row is the pixel location in the front image (Z=0).
        back_image: Each row is the pixel location in the back image (Z=150).
    Returns:
        rms_error: The root mean square error of reprojecting the points back
            into the images.
    """
    img_num1 = front_image.shape[0]
    img_num2 = back_image.shape[0]

    x = np.zeros((2, img_num1+img_num2))
    for i in xrange(img_num1):
        x[:, i] = front_image[i, :].T
    for j in xrange(img_num2):
        x[:, j + img_num1] = back_image[j, :].T
    x_ones = np.ones((1, x.shape[1]))
    x = np.vstack((x, x_ones))

    X = np.zeros((2, img_num1+img_num2))
    for i in xrange(img_num1):
        X[:, i] = real_XY[i, :].T
    for j in xrange(img_num2):
        X[:, j + img_num1] = real_XY[j, :].T
    Z = np.zeros((1, X.shape[1]))
    for k in xrange(Z.shape[1]):
        if k >= img_num1:
            Z[:, k] = 150
    X = np.vstack((X, Z))
    X_ones = np.ones((1, X.shape[1]))
    X = np.vstack((X, X_ones))

    x_pred = camera_matrix.dot(X)
    diff_sqr = (x_pred - x) ** 2
    diff_sum = np.sum(np.sum(diff_sqr, axis=0))
    diff_sum /= (img_num1 + img_num2)
    rms_error = np.sqrt(diff_sum)
    return rms_error

if __name__ == '__main__':
    # Load the example coordinates setup.
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    rmse = rms_error(camera_matrix, real_XY, front_image, back_image)
    print "Camera Matrix:\n", camera_matrix
    print
    print "RMS Error: ", rmse

