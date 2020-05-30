import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sp
# This import is mandatory for plotting 3D surfaces
from mpl_toolkits import mplot3d
import cv2 as cv


# Q4
def Gaussian_2D(size, sigma):
    """
    Generate a gaussian kernel with respect to size and sigma
    :param size:    size of the Gaussian kernel
    :param sigma:   sigma of the Gaussian kernel
    :return:        the normalized Gaussian kernel
    """
    if size % 2 == 0:
        print("Size of the kernel must be a odd number")
        return None
    else:

        # Calculate the kernel with gaussian function.
        values = np.arange(-(size - 1) / 2, (size - 1) / 2 + 1, 1)
        constant = 1 / (2 * np.pi * (sigma ** 2))
        y_vector = (np.e ** -(np.square(values.transpose()) / (2 * sigma ** 2))).reshape(size, -1)
        x_vector = (np.e ** -(np.square(values) / (2 * sigma ** 2))).reshape((-1, size))
        kernel = constant * np.matmul(y_vector, x_vector)
        # Normalize the kernel
        normalized_kernel = kernel / np.sum(kernel)
        return normalized_kernel


# Q5
def LoG(size, sigma):
    """
    Generate a LoG kernel with respect to size and sigma
    :param size:    size of the LoG kernel
    :param sigma:   sigma of the LoG kernel
    :return:        the normalized LoG kernel
    """
    if size % 2 == 0:
        print("Size of the kernel must be a odd number")
        return None
    else:
        kernel = np.empty((size, size), dtype=np.float)

        # Calculate the kernel with LoG formula.
        for y in range(0, size):
            for x in range(0, size):
                norm_x = x - (size - 1) / 2
                norm_y = y - (size - 1) / 2
                factor = - (np.square(norm_x) + np.square(norm_y)) / (2 * np.square(sigma))
                constant = (1 / (np.pi * sigma ** 4)) * (1 + factor)
                kernel[y, x] = constant * (np.e ** factor)

        # Return the normalized kernel
        return kernel / np.sum(kernel)


# Q6
def LoG_convolution(img, size, sigma):
    """
    Convolve LoG with the image chosen
    :param img:     name of the image
    :param size:    size of the Log kernel
    :param sigma:   sigma of the Log kernel
    :return:        result of the convolution
    """
    image = cv.imread(img)
    greyscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    result = sp.convolve2d(greyscale, LoG(size, sigma), mode='same', boundary='fill', fillvalue=0)
    cv.imwrite(img[0:-4] + "_LoG_convolution_result.png", result)
    return result


# Q7
def Zero_crossing(input, img_name, threshold):
    """
    Detect the zero-crossing of on an image
    :param input:       The image matrix
    :param img_name:    The name of the image
    :param threshold:   Threshold for suppressing zero-crossings
    """
    orginal_img = cv.imread(img_name)
    copy1 = np.zeros((orginal_img.shape[0], orginal_img.shape[1]))

    # Detect the zero crossing on y direction
    for y in range(0, input.shape[0]):
        zero_crossings_y = np.where(np.diff(np.signbit(input[y, :])))[0]

        # Mark the point of zero-crossing on y direction with yellow on original image
        orginal_img[y, zero_crossings_y, :] = np.array([0, 255, 255])

        # Calculate the strength of zero-crossings on the y direction and save them to a different matrix.
        for i in zero_crossings_y:
            s = np.abs(input[y, i] - input[y, i + 1])
            copy1[y, i] = s

    # Detect the zero crossing on y direction
    for x in range(0, input.shape[1]):
        zero_crossings_x = np.where(np.diff(np.signbit(input[:, x])))[0]

        # Mark the point of zero-crossing on y direction with yellow on original image
        orginal_img[zero_crossings_x, x, :] = np.array([0, 255, 255])

        # Calculate the strength of zero-crossings on the y direction and save them to a different matrix
        for j in zero_crossings_x:
            s = np.abs(input[j, x] - input[j + 1, x])
            copy1[j, x] = s

    temp = (copy1 / np.max(copy1)) * 255
    # Suppress small zero crossings
    copy1 = np.where(temp < threshold, 0, 255)
    # Save image.
    cv.imwrite(img_name[0:-4] + '_Edge_overlay.png', orginal_img)
    cv.imwrite(img_name[0:-4] + '_edge_normalize_suppressed.png', copy1)


def plot_gaussian(size, sigma):
    """
    Use the Gaussian_2D() function to create and plot the kernel as surfaces in 3D
    :param size:    Size of the Gaussian kernel
    :param sigma:   Sigma of the Gaussian kernel
    :return:
    """
    z = Gaussian_2D(size, sigma)
    x = np.arange(-1 * (size - 1) / 2, ((size - 1) / 2) + 1, 1)
    y = np.arange(-1 * (size - 1) / 2, ((size - 1) / 2) + 1, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('Gaussian_2D_s' + str(size) + '_sig' + str(sigma) + '.png')


def plot_LoG(size, sigma):
    """
    Use the LoG() function to create and plot the kernel as surfaces in 3D
    :param size:    Size of the LoG kernel
    :param sigma:   Sigma of the LoG kernel
    """
    z = LoG(size, sigma)
    x = np.arange(-1 * (size - 1) / 2, ((size - 1) / 2) + 1, 1)
    y = np.arange(-1 * (size - 1) / 2, ((size - 1) / 2) + 1, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('LoG_2D_s' + str(size) + '_sig' + str(sigma) + '.png')


if __name__ == "__main__":
    Zero_crossing(LoG_convolution('1.jpg', 7, 1), '1.jpg', 15)
    Zero_crossing(LoG_convolution('Paolina.jpg', 7, 1), 'Paolina.jpg', 15)

    plot_gaussian(15, 2)
    plot_gaussian(21, 3)
    plot_LoG(15, 2)
    plot_LoG(21, 3)
