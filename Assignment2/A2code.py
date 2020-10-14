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

def Resize_to_half_and_tenth(imageName):
    """
    Create image at half and 1/10 the original scale
    :param imageName: Name of the image
    :return: None
    """
    image = cv.imread(imageName)
    half = cv.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
    tenth = cv.resize(image, (int(image.shape[1]/10), int(image.shape[0]/10)))
    cv.imwrite(imageName[0:-4]+'_half.png', half)
    cv.imwrite(imageName[0:-4]+'_1_tenth.png', tenth)

def Harris_Corner_Detector(imgName, window_size, threshold, a=0.06, smooth_image=False, saveImage=True):
    """
    Produce a image where detected connors are circled in red
    :param imgName:     Name of the target image
    :param window_size: Patch size of the image
    :param threshold:   Threshold for R
    :param a:           value of a
    :param smooth_image: Option for smooth the image or not
    :param saveImage:    Option for saving image or not
    :return:            None
    """
    # Obtain grey scale image

    colour = cv.imread(imgName)
    image = cv.cvtColor(colour, cv.COLOR_RGB2GRAY)
    if smooth_image:
        image = cv.GaussianBlur(image, (window_size, window_size), 1)
    # Calculate the gradient
    dy = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    dx = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)

    # Calculate moment matrix components
    dxdy = cv.GaussianBlur(dx*dy, (window_size, window_size), 1)
    dx_2 = cv.GaussianBlur(dx*dx, (window_size, window_size), 1)
    dy_2 = cv.GaussianBlur(dy*dy, (window_size, window_size), 1)
    R_field = np.empty((dx.shape[0], dx.shape[1]), dtype=np.float64)

    for y in range(9, dx.shape[0]):
        for x in range(0, dx.shape[1]):
            M = np.array([[dx_2[y, x], dxdy[y, x]],
                          [dxdy[y, x], dy_2[y, x]]])

            # Calculate eigenvalues and traces
            eigen_values = np.linalg.eigvals(M)
            R = dx_2[y, x]*dy_2[y, x] - dxdy[y, x]**2 - a * (np.sum(eigen_values) ** 2)
            R_field[y, x] = R
    # Calculate the percentiles
    percentile = np.percentile(R_field, threshold)
    for y in range(9, dx.shape[0]):
        for x in range(0, dx.shape[1]):
            # Use percentile to screen out weak corners
            if R_field[y][x] > percentile and saveImage:
                cv.circle(colour, (x, y), 1, (0, 0, 255))
    if saveImage:
        cv.imwrite(imgName[0:-4] + '_t' + str(threshold)+'_' + str(smooth_image) + '_' + 'cornor_detect.png', colour)
    return R_field

def Gaussian_pyramid(imgName):
    """
    Generate Gaussian pyramids
    :param imgName: name of the image
    :return: Different levels of the pyramid
    """
    targetImage = cv.imread(imgName)
    greyScale = cv.cvtColor(targetImage, cv.COLOR_RGB2GRAY)
    pyramids = []

    # Generate levels of the pyramid
    for i in range(0, 7):
        kernel = Gaussian_2D(5,2**i)
        blur = cv.GaussianBlur(greyScale, (5, 5), 2**i)
        resized_image = cv.resize(blur, (int(blur.shape[0]/(2**i)), int(blur.shape[1]/(2**i))))
        pyramids.append(resized_image)

    # Composite the image
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
    """
    Generate Laplacian pyramids
    :param imgName: Name of the image
    :return:        Different levels of the pyramid
    """
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

    # Create levels for the laplacian pyramids
    for i in range(0, 6):
        upsample = cv.resize(pyramids[i + 1], (int(pyramids[i + 1].shape[0]) * 2, int(pyramids[i + 1].shape[1]) * 2))
        DoGs.append(pyramids[i] - upsample)
        normalized_DoGs.append(cv.normalize(pyramids[i] - upsample, None, 0 ,255))

    # Generate composite image
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

    # Generate composite image for normalized laplacian
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

# Q7
def local_max(original_img, threshold):
    """
    This function generate keypoints for Question 7
    :param original_img: Name of the image
    :param threshold:    Threshold for detecting strong extrema
    """
    colour = cv.imread(original_img)
    LoGs = Laplacian_pyramid(original_img)
    position_vectors = []
    # Colors use for drawing circles
    draw_colour = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]

    for i in range(0, 5):
        neighbors = None
        rows = LoGs[i].shape[0]
        col = LoGs[i].shape[1]
        # Scale the neighbour pyramid level to the same size as the current level and put them into a 3 dimensional
        # array for the convince of determine extrema
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
                # make sure the index does not go out of bounds
                y_coord = np.clip([y - 1, y + 2], 0, rows)
                x_coord = np.clip([x - 1, x + 2], 0, col)
                region = neighbors[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :]

                #
                if np.max(region) == value or np.min(region) == value:
                    temp1 = -1 * region + value
                    temp2 = -1 * region + value
                    temp1[y - y_coord[0], x - x_coord[0], 0] = threshold
                    temp2[y - y_coord[0], x - x_coord[0], 0] = -threshold
                    if np.min(temp1) >= threshold or np.max(temp2) <= -threshold:
                        position_vectors.append([y*2**i, x*2**i, i])

    for v in position_vectors:
        cv.circle(colour, (v[1], v[0]), 3, draw_colour[v[-1]])
    cv.imwrite(original_img[0:-4] + "_localmax.png", colour)




if __name__ =="__main__":
    # Q1_plot()
    # Resize_to_half_and_tenth('shapes.png')
    # Resize_to_half_and_tenth('cnTower.jpg')
    # Harris_Corner_Detector('shapes.png', 5, 99, smooth_image=False)
    # Harris_Corner_Detector('shapes.png', 5, 99, smooth_image=True)
    # Harris_Corner_Detector('shapes_half.png', 5, 99, smooth_image=True)
    # Harris_Corner_Detector('shapes_1_tenth.png', 5, 99, smooth_image=True)
    # Harris_Corner_Detector('cnTower.jpg', 5, 99, smooth_image=True)
    # Harris_Corner_Detector('cnTower_half.png', 5, 99, smooth_image=True)
    # Harris_Corner_Detector('cnTower_1_tenth.png', 5, 99, smooth_image=True)
    # Gaussian_pyramid('sunflower.jpg')
    # Laplacian_pyramid('sunflower.jpg')
    local_max("sunflower.jpg", 1)
