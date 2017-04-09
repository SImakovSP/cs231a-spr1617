# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg')
    img2 = misc.imread('image2.jpg')

    plt.subplot(1, 2, 1)
    plt.title("original image1")
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title("original image2")
    plt.imshow(img2)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    def imagescaling(img):
        pixmax = np.max(img)
        pixmin = np.min(img)
        return (img - pixmin) / (pixmax - pixmin)

    # BEGIN YOUR CODE HERE
    img1 = img1.astype('float')
    img2 = img2.astype('float')

    # scale images
    img1 = imagescaling(img1)
    img2 = imagescaling(img2)

    plt.subplot(1, 2, 1)
    plt.title("image1 w/ double precision")
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title("image2 w/ double precision")
    plt.imshow(img2)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    img12 = img1 + img2
    img12 = imagescaling(img12)
    plt.title("(3c) Summation of Two Images")
    plt.imshow(img12)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    # image1 and image2 have the same size
    imageShape = np.shape(img1)
    row = np.shape(img1)[0]
    col = np.shape(img1)[1]
    centerCol = np.shape(img1)[1]/2
    filterL = np.ones((row, centerCol, 3))
    filterR = np.zeros((row, col - centerCol, 3))
    filter1 = np.concatenate((filterL, filterR), axis=1)
    filter2 = np.concatenate((filterR, filterL), axis=1)

    newImage1 = np.multiply(img1, filter1) + np.multiply(img2, filter2)
    plt.title("(3d) Half-half Image")
    plt.imshow(newImage1)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros(imageShape)
    for i in range(row):
        if i % 2 == 0:
            newImage2[i] = img2[i] # even numbered row from image2
        else:
            newImage2[i] = img1[i] # odd numbered row from image1

    plt.title("(3e) Image w/ different rows from image 1&2")
    plt.imshow(newImage2)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    evenRowMask = np.concatenate((np.ones((1, col, 3)), np.zeros((1, col, 3))), axis=0)
    oddRowMask = np.concatenate((np.zeros((1, col, 3)), np.ones((1, col, 3))), axis=0)
    rowFilter1 = np.tile(oddRowMask, (row/2, 1, 1))
    rowFilter2 = np.tile(evenRowMask, (row/2, 1, 1))

    newImage3 = np.multiply(img1, rowFilter1) + np.multiply(img2, rowFilter2)
    plt.title("(3f) Image w/ different rows from image 1&2")
    plt.imshow(newImage3)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    def rgb2gray(pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

    # Convert the image
    newImage4 = np.zeros((newImage3.shape[0], newImage3.shape[1])) # init 2D numpy array
    for r in range(len(newImage3)):
       for c in range(len(newImage3[r])):
          newImage4[r][c] = rgb2gray(newImage3[r][c])

    plt.title("(3g) Grayscale Image of (3f)")
    plt.imshow(newImage4, cmap=plt.cm.Greys_r)
    plt.show()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
