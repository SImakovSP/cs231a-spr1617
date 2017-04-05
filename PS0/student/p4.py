# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg', 'L')

    # Reconstruction based on reduced SVD
    U, s, V = np.linalg.svd(img1, full_matrices=False)
    S = np.diag(s)
    imgSVD = np.dot(U, np.dot(S, V))

    plt.subplot(1, 2, 1)
    plt.title("original image1")
    plt.imshow(img1, cmap=plt.cm.Greys_r)

    plt.subplot(1, 2, 2)
    plt.title("reconstruction based on SVD")
    plt.imshow(imgSVD, cmap=plt.cm.Greys_r)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    row = np.shape(img1)[0]
    col = np.shape(img1)[1]
    u1 = np.reshape(U[:, 0], (row, 1))
    s1= S[0, 0]
    v1 = np.reshape(V[0, :], (1, col))
    rank1approx = np.dot(u1, s1 * v1)
    plt.title("(4b) best rank 1 approximation of image1")
    plt.imshow(rank1approx, cmap=plt.cm.Greys_r)
    plt.show()
    misc.imsave('rank1approx.jpg', rank1approx)
    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE\
    rank20approx = np.zeros(np.shape(img1))
    for i in range(20):
        ui = np.reshape(U[:, i], (row, 1))
        si= S[i, i]
        vi = np.reshape(V[i, :], (1, col))
        rank20approx += np.dot(ui, si * vi)

    plt.title("(4c) best rank 20 approximation of image1")
    plt.imshow(rank20approx, cmap=plt.cm.Greys_r)
    plt.show()
    misc.imsave('rank20approx.jpg', rank20approx)
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
