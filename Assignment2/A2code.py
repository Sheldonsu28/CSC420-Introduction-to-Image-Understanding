import numpy as np
import cv2 as cv
import scipy.signal as sp
import matplotlib.pyplot as plt

def Q1_plot():
    """
    Generate the plot for Q1
    """
    x = np.array([-2, -1.6, -1.3, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.3, 1.6, 2])
    y = np.array([4, 2.8, 1.9, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1.9, 2.8, 4])
    plt.stem(x, y, linefmt='blue', markerfmt=" ", basefmt=" ", use_line_collection=True)
    plt.title("Interpolated signals")
    plt.savefig("Q1.png")
    plt.show()
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


def Harris_Corner_Detector(imgName, window_size, threshold, a=0.06, supply_image=None, smooth=True, saveImage=True):
    """
    Produce a image where detected connors are circled in red
    :param imgName:     Name of the target image
    :param window_size: Patch size of the image
    :param threshold:   Threshold for R
    :param a:           value of a
    :return:            None
    """
    # Obtain grey scale image
    if supply_image is None:
        colour = cv.imread(imgName)
        image = cv.cvtColor(colour, cv.COLOR_RGB2GRAY)
    else:
        colour = supply_image
        image = supply_image
    # Calculate the gradient
    if smooth:
        image = cv.GaussianBlur(cv.cvtColor(colour, cv.COLOR_RGB2GRAY), (5, 5), 1)
    Gaussian = Gaussian_2D(window_size, 1)
    # Gaussian = np.ones((window_size, window_size))
    gradients = np.gradient(image)
    dy = gradients[0]
    dx = gradients[1]

    # Generate a gaussian kernel

    dxdy = sp.convolve2d(dx*dy, Gaussian, mode='same', boundary='fill', fillvalue=0)
    dx_2 = sp.convolve2d(np.square(dx), Gaussian, mode='same', boundary='fill', fillvalue=0)
    dy_2 = sp.convolve2d(np.square(dy), Gaussian, mode='same', boundary='fill', fillvalue=0)
    R_field = np.empty((dx.shape[0], dx.shape[1]), dtype=np.float64)
    for y in range(9, dx.shape[0]):
        for x in range(0, dx.shape[1]):
            M = np.array([[dx_2[y, x], dxdy[y, x]],
                          [dxdy[y, x], dy_2[y, x]]])

            eigen_values = np.linalg.eigvals(M)
            R = eigen_values[0] * eigen_values[1] - a * (np.sum(eigen_values) ** 2)
            R_field[y, x] = R

    percentile = np.percentile(R_field, threshold)
    for y in range(9, dx.shape[0]):
        for x in range(0, dx.shape[1]):
            if R_field[y][x] > percentile and saveImage:
                cv.circle(colour, (x, y), 1, (0, 0, 255))

    if saveImage:
        cv.imwrite(imgName[0:-4] + '_t' + str(threshold) + '_' + 'cornor_detect.png', colour)
    return R_field

def Gaussian_pyramid(imgName):
    targetImage = cv.imread(imgName)
    greyScale = cv.cvtColor(targetImage, cv.COLOR_RGB2GRAY)
    pyramids = []
    for i in range(0, 7):
        kernel = Gaussian_2D(5,2**i)
        blur = sp.convolve2d(greyScale, kernel, mode='same')
        resized_image = cv.resize(blur, (int(blur.shape[0]/(2**i)), int(blur.shape[1]/(2**i))))
        pyramids.append(resized_image)

    composite = np.zeros((greyScale.shape[0], greyScale.shape[1] + int(greyScale.shape[1]/2)), dtype=int)
    composite[:greyScale.shape[0], :greyScale.shape[1]] = pyramids[0]

    row = 0
    col = greyScale.shape[1]
    for p in pyramids[1:]:
        y = p.shape[0]
        x = p.shape[1]
        composite[row:row + y, col:col + x] = p
        row += y
    cv.imwrite("Gaussian_pyramid.png", composite)
    return pyramids

def Laplacian_pyramid(imgName):
    targetImage = cv.imread(imgName)
    greyScale = cv.cvtColor(targetImage, cv.COLOR_RGB2GRAY)
    pyramids = []
    for i in range(0, 7):
        kernel = Gaussian_2D(5, 2**i)
        blur = sp.convolve2d(greyScale, kernel, mode='same')
        resized_image = cv.resize(blur, (int(blur.shape[0]/(2**i)), int(blur.shape[1]/(2**i))))
        pyramids.append(resized_image)
    DoGs = []
    normalized_DoGs = []
    for i in range(0, 6):
        upsample = cv.resize(pyramids[i + 1], (int(pyramids[i + 1].shape[0]) * 2, int(pyramids[i + 1].shape[1]) * 2))
        DoGs.append(pyramids[i] - upsample)
        normalized_DoGs.append((pyramids[i] - upsample)*255/np.max((pyramids[i] - upsample)))

    composite = np.zeros((greyScale.shape[0], greyScale.shape[1] + int(greyScale.shape[1] / 2)), dtype=int)
    composite[:greyScale.shape[0], :greyScale.shape[1]] = DoGs[0]
    composite_normalized = np.zeros((greyScale.shape[0], greyScale.shape[1] + int(greyScale.shape[1] / 2)), dtype=int)
    composite_normalized[:greyScale.shape[0], :greyScale.shape[1]] = normalized_DoGs[0]

    row = 0
    col = greyScale.shape[1]
    for p in DoGs[1:]:
        y = p.shape[0]
        x = p.shape[1]
        composite[row:row + y, col:col + x] = p
        row += y

    row = 0
    col = greyScale.shape[1]
    for p in DoGs[1:]:
        y = p.shape[0]
        x = p.shape[1]
        composite_normalized[row:row + y, col:col + x] = p
        row += y

    cv.imwrite("Laplacian_pyramid.png", composite)
    cv.imwrite("Laplacian_pyramid_normalized.png", composite_normalized)
    return DoGs

def local_max(original_img, threshold):
    colour = cv.imread(original_img)
    LoGs = Laplacian_pyramid(original_img)
    position_vectors = []
    draw_colour = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255)]
    for i in range(0, 5):
        neighbors = None
        rows = LoGs[i].shape[0]
        col = LoGs[i].shape[1]
        if i == 0:
            neighbors = np.empty((rows, col, 2))
            neighbors[:, :, 0] = LoGs[i]
            neighbors[:, :, 1] = cv.resize(LoGs[1], (LoGs[1].shape[0] * 2, LoGs[1].shape[1] * 2))
        else:
            neighbors = np.empty((rows, col, 3))
            neighbors[:, :, 0] = LoGs[i]
            neighbors[:, :, 1] = cv.resize(LoGs[i - 1], (int(LoGs[i - 1].shape[0]/2), int(LoGs[i - 1].shape[1]/2)))
            neighbors[:, :, 2] = cv.resize(LoGs[i + 1], (int(LoGs[i + 1].shape[0]*2), int(LoGs[i + 1].shape[1] * 2)))

        for y in range(0, rows):
            for x in range(0, col):

                value = neighbors[y, x, 0]
                y_coord = np.clip([y - 1, y + 2], 0, rows)
                x_coord = np.clip([x - 1, x + 2], 0, col)
                region = neighbors[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :]

                if np.max(region) == value:
                    temp = -1 * region + value
                    temp[y - y_coord[0], x - x_coord[0], 0] = threshold
                    if np.min(temp) >= threshold:
                        position_vectors.append([y*2**i, x*2**i, i])

    for v in position_vectors:
        cv.circle(colour, (v[1], v[0]), 3, draw_colour[v[-1]])
    cv.imwrite(original_img[0:-4] + "_localmax.png", colour)




if __name__ =="__main__":
    Q1_plot()
    # Harris_Corner_Detector('shapes.png', 5, 1e5,smooth=True)
    # Harris_Corner_Detector('shapes_half.png', 15, 1e5,smooth=True)
    # Harris_Corner_Detector('shapes_1_tenth.png', 5, 1e5,smooth=True)
    # Harris_Corner_Detector('shapes.png', 5, 1e5, smooth=True)
    # Harris_Corner_Detector('cnTower.jpg', 5, 99.5, smooth=True)
    # Harris_Corner_Detector('cnTower_half.png', 5, 1e5,smooth=True)
    # Harris_Corner_Detector('cnTower_1_tenth.png', 5, 1e5,smooth=True)
    # Gaussian_pyramid('sunflower.jpg')
    # Laplacian_pyramid('sunflower.jpg')
    # local_max("sunflower.jpg", 3)
