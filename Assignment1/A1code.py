import numpy as np
from scipy import signal as sp
import cv2 as cv

#Q4
def Guassian_2D(size, sigma):
    if size%2 == 0:
        print("Size of the kernel must be a odd number")
        return None
    else:

        values = np.arange(-(size - 1)/2, (size - 1)/2 + 1, 1)
        constant = 1/(2*np.pi*(sigma**2))
        y_vector = (np.e**(np.square(values.transpose())/(2*sigma**2))).reshape(size,-1)
        x_vector = (np.e**(np.square(values)/(2*sigma**2))).reshape((-1, size))
        kernel = constant * np.matmul(y_vector, x_vector)
        normalized_kernel = kernel / np.sum(kernel)
        return normalized_kernel

#Q5
def LoG(size, sigma):
    if size % 2 == 0:
        print("Size of the kernel must be a odd number")
        return None
    else:
        kernel = np.empty((size, size), dtype=np.float)

        for y in range(0, size):
            for x in range(0, size):
                norm_x = x - (size - 1)/2
                norm_y = y - (size - 1)/2
                factor = - (np.square(norm_x) + np.square(norm_y))/(2*np.square(sigma))
                constant = (np.square(norm_x) + np.square(norm_y) - 2*np.square(sigma))/(sigma**4)
                kernel[y, x] = constant * (np.e**factor)

        return kernel/np.sum(kernel)

#Q6
def LoG_convolution(image):
    image = cv.imread(image)
    greyscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    result = sp.convolve2d(greyscale, LoG(7, 1), mode='same',boundary='fill', fillvalue=0)
    cv.imshow('hi', result)
    cv.imwrite('LogConvolutuib.png', result)
    return result

#Q7
def Zero_crossing(input):
    sign_matrix = np.diff(np.sign(input))
    new = np.where(sign_matrix >= 0, 255, 0 )

    cv.imwrite('zeroCrossing.png', new)

if __name__ =="__main__":
    Zero_crossing(LoG_convolution('Paolina.jpg'))
