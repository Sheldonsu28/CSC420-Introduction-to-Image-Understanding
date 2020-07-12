import numpy as np
import cv2 as cv
from scipy import ndimage

def Gaussian_2D(size, sigma):
    """
    Generate a gaussian kernel with respect to size and sigma
    :param size:    size of the Gaussian kernel
    :param sigma:   sigma of the Gaussian kernel
    :return:        the normalized Gaussian kernel
    """


    # Calculate the kernel with gaussian function.
    values = np.arange(-(size - 1) / 2, (size - 1) / 2 + 1, 1)
    constant = 1 / (2 * np.pi * (sigma ** 2))
    y_vector = (np.e ** -(np.square(values.transpose()) / (2 * sigma ** 2))).reshape(size, -1)
    x_vector = (np.e ** -(np.square(values) / (2 * sigma ** 2))).reshape((-1, size))
    kernel = constant * np.matmul(y_vector, x_vector)
    # Normalize the kernel
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel


def gaussian_pyramid(img, downscale=2, num_layer=7):
    layers = [img]
    for i in range(1, num_layer):
        smoothed = ndimage.gaussian_filter(img, downscale ** i)
        downsampled = cv.resize(smoothed, None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv.INTER_AREA)
        layers.append(downsampled)
    return layers



def Laplacian_pyramid_and_Gaussain(imgName):
    """
    Generate Laplacian pyramids
    :param imgName: Name of the image
    :return:        Different levels of the pyramid
    """
    targetImage = cv.imread(imgName)
    greyScale = cv.cvtColor(targetImage, cv.COLOR_RGB2GRAY)
    pyramids = [greyScale]
    for i in range(1, 7):
        blur = ndimage.gaussian_filter(greyScale, 2 ** i)
        resized_image = cv.resize(blur, None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv.INTER_AREA)
        pyramids.append(resized_image)
    DoGs = []

    # Create levels for the laplacian pyramids
    for i in range(0, len(pyramids) - 1):
        upsampled = cv.resize(pyramids[i + 1], None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
        diff = cv.subtract(pyramids[i], upsampled)
        cv.normalize(diff, diff, 0, 255, cv.NORM_MINMAX)
        DoGs.append(diff)

    return DoGs, pyramids


def local_max(original_img, threshold, kernel=5):
    """
    This function generate keypoints for Question 7
    :param original_img: Name of the image
    :param threshold:    Threshold for detecting strong extrema
    """
    colour = cv.imread(original_img)
    grey = cv.imread(original_img, cv.IMREAD_GRAYSCALE)
    LoGs, Gaussian_pyramids = Laplacian_pyramid_and_Gaussain(original_img)
    position_vectors = []
    # Colors use for drawing circles
    draw_colour = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    neighbor = kernel // 2

    for i in range(len(LoGs) - 1):
        neighbors = None
        rows = LoGs[i].shape[0]
        col = LoGs[i].shape[1]
        # Scale the neighbour pyramid level to the same size as the current level and put them into a 3 dimensional
        # array for the convince of determine extrema
        if i == 0:
            neighbors = np.empty((rows, col, 2))
            neighbors[:, :, 0] = LoGs[i]
            neighbors[:, :, 1] = cv.resize(LoGs[1], None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
        else:
            neighbors = np.empty((rows, col, 3))
            neighbors[:, :, 0] = LoGs[i]
            neighbors[:, :, 1] = cv.resize(LoGs[i - 1], None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
            neighbors[:, :, 2] = cv.resize(LoGs[i + 1], None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)

        for y in range(0, rows):
            for x in range(0, col):

                value = neighbors[y, x, 0]
                # make sure the index does not go out of bounds
                y_coord = np.clip([y - neighbor, y + neighbor + 1], 0, rows)
                x_coord = np.clip([x - neighbor, x + neighbor + 1], 0, col)
                region = neighbors[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :]

                diff1 = value - region
                diff2 = region - value
                _, _, z = diff1.shape
                if (region.max() == value and len(diff1[diff1 >= threshold]) >= z * np.square(kernel) - 1) \
                        or (region.min() == value and len(diff2[diff2 >= threshold]) >= z * np.square(kernel) - 1):
                    position_vectors.append([y, x, i])

        for v in position_vectors:
            cv.circle(colour, (v[1] * 2 ** v[-1], v[0] * 2 ** v[-1]), 3, draw_colour[v[-1]])
    cv.imwrite(original_img[0:-4] + "_localmax.png", colour)
    return [LoGs, Gaussian_pyramids, position_vectors]
