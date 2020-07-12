import numpy as np
import scipy as sp
import cv2 as cv
from scipy import signal as sps
from matplotlib import pyplot as plt
from Assignment3 import A2Q7 as q7
from scipy.io import loadmat

# Code for Q1
def Q1(imgName, threshlod,kernel=3):

    preprocssing = q7.local_max(imgName, threshlod, kernel)
    Gaussain = preprocssing[1]
    position_vectors = preprocssing[2]
    Gaussain_gradient_x = []
    Gaussain_gradient_y = []
    magnitude = []
    orientation = []
    return_value = []
    selected_vectors = []
    kernel = q7.Gaussian_2D(16, 1)
    for level in Gaussain:
        grad_x, grad_y = cv.Sobel(level, cv.CV_64F, 1, 0), cv.Sobel(level, cv.CV_64F, 0, 1)
        Gaussain_gradient_x.append(grad_x)
        Gaussain_gradient_y.append(grad_y)
        magnitude.append(np.sqrt(np.square(grad_x) + np.square(grad_y)))
        orientation.append(np.arctan2(grad_y, grad_x))

    for v in position_vectors:
        img_shape = Gaussain[v[-1]].shape
        if 16 <= v[0] <= img_shape[0] - 17 and 16 <= v[1] <= img_shape[0] - 17:
            y_coordinate = np.clip([v[0] - 8, v[0] + 8], 0, img_shape[0] - 1)
            x_coordinate = np.clip([v[1] - 8, v[1] + 8], 0, img_shape[1] - 1)
            gradient_magnitude = magnitude[v[-1]][y_coordinate[0]:y_coordinate[1], x_coordinate[0]:x_coordinate[1]]
            orientations = orientation[v[-1]][y_coordinate[0]:y_coordinate[1], x_coordinate[0]:x_coordinate[1]]
            weighted_magnitude = gradient_magnitude*kernel
            return_value.append([orientations, weighted_magnitude, v])
            selected_vectors.append(v)

    # x_coord = (selected_vectors[0][0] - 8, selected_vectors[0][0] + 8)
    # y_coord = (selected_vectors[0][1] - 8, selected_vectors[0][1] + 8)
    #
    # X, Y = np.meshgrid(np.arange(-8, 8), np.arange(-8, 8))
    # U = Gaussain_gradient_x[selected_vectors[0][-1]][x_coord[0]:x_coord[1], y_coord[0]:y_coord[1]]
    # V = Gaussain_gradient_y[selected_vectors[0][-1]][x_coord[0]:x_coord[1], y_coord[0]:y_coord[1]]

    # plt.quiver(X, Y, U, V, pivot='mid')
    # plt.title("Image gradient orientation")
    # plt.legend()
    # plt.savefig(imgName[0:-4] + "_orientation.png")
    return return_value, position_vectors

def Q2(data):
    values = data[0]
    return_list = []
    for v in values:
        flatten_orientation = np.ndarray.flatten(v[0]) * 180/np.pi
        flatten_weighted_magnitude = np.ndarray.flatten(v[1])
        location_vector = v[-1]
        counts = np.zeros(36)
        for i in range(0, 36):
            counts[i] = sum(flatten_weighted_magnitude[(flatten_orientation >= i*10) & (flatten_orientation < 10*(i+1))])
        peak_index = np.argmax(counts)
        for i in range(0, 36):
            location_vector.append(counts[int((peak_index+i) % 36)])
        return_list.append(np.array(location_vector))
        # if v is values[0]:
        #     plt.hist(flatten_orientation, bins=np.arange(0, 361, 10))
        #     plt.show()
    return return_list

def Q3(image, x0, y0, theta, s):
    img = cv.imread(image)
    rotation_matrix = cv.getRotationMatrix2D((y0, x0), theta, s)
    result = cv.warpAffine(img, rotation_matrix, img.shape[0:2])
    cv.imwrite(image[0:-4] + "_rotated.png", result)

def Bhattacharyya_coefficient(vector1, vector2):
    vector1 = vector1/np.sqrt(np.sum(vector1**2))
    vector2 = vector2/np.sqrt(np.sum(vector2**2))
    BC = np.sum(np.sqrt(vector1 * vector2))
    return BC

def Q4(originalImage, rotatedImage, original_vectors, rotated_vectors, rotation_param, window_size):
    orig_image = cv.imread(originalImage)
    rot_image = cv.imread(rotatedImage)
    output = np.empty((orig_image.shape[0], 2*orig_image.shape[1], orig_image.shape[-1]))
    output[:orig_image.shape[0], :orig_image.shape[1], :] = orig_image[:, :, :]
    output[:orig_image.shape[0], orig_image.shape[1]:, :] = rot_image[:, :, :]
    rotation_matrix = cv.getRotationMatrix2D(rotation_param[0], rotation_param[1], rotation_param[2])
    matched_vectors = []
    for v in original_vectors:
        new_point = rotation_matrix @ np.array([[v[1]], [v[0]], [1]])
        new_x, new_y = int(new_point[0]), int(new_point[1])
        min_value = np.inf
        min_v = None
        for rv in rotated_vectors:
            if abs(rv[0] - new_y) <= window_size and abs(rv[1] - new_x) <= window_size:
                value = np.sum(np.abs(v[3:] - rv[3:]))
                if value < min_value:
                    min_value = value
                    min_v = rv
        if min_v is not None:
            matched_vectors.append((v[:3], min_v[:3]))

    shape = orig_image.shape
    print("Draw!")
    for m in matched_vectors:
        value = m
        src = value[1]
        target = value[0]
        point1 = (int(shape[1] + src[1]*2**src[-1]), int(src[0]*2**src[-1]))
        point2 = (int(target[1]*2**target[-1]), int(target[0]*2**target[-1]))
        cv.line(output, point1, point2, (255, 0, 0))
    cv.imwrite(rotatedImage[0:-4]+"_match.png", output)

def Q7():
    sift_features = loadmat('sift_features.mat')
    print(sift_features.keys())
    features_1 = sift_features["features_1"]
    keypoints_1 = sift_features["keypoints_1"]
    features_2 = sift_features["features_2"]
    keypoints_2 = sift_features["keypoints_2"]
    theta = sift_features["theta"]
    print(keypoints_1[:, 0])
    # print(len(features_1[:, 0]))
    # print(theta)

def Q8(img1, img2, feature_num, threshlod=0.5):
    orig_img1 = cv.imread(img1)
    greyScale1 = cv.cvtColor(orig_img1, cv.COLOR_RGB2GRAY)
    orig_img2 = cv.imread(img2)
    greyScale2 = cv.cvtColor(orig_img2, cv.COLOR_RGB2GRAY)

    sift = cv.xfeatures2d.SIFT_create(nfeatures=feature_num)

    kps1, descriptor1 = sift.detectAndCompute(greyScale1, None)
    kps2, descriptor2 = sift.detectAndCompute(greyScale2, None)

    orig_img1 = cv.drawKeypoints(orig_img1, kps1, None)
    orig_img2 = cv.drawKeypoints(orig_img2, kps2, None)
    cv.imwrite(img1[:-3]+'sift_detect.jpg', orig_img1)
    cv.imwrite(img2[:-3] + 'sift_detect.jpg', orig_img2)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    match_1 = []
    for m, n in matches:
        if m.distance < threshlod*n.distance:
            match_1.append([m])
    output = cv.drawMatchesKnn(orig_img1, kps1, orig_img2, kps2, match_1, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("Q8_match.png", output)









if __name__ =="__main__":
    # # Q1Values = Q1("UofT.jpg", 30, 5)
    # # print(len(Q1Values[0]))
    # # feature_vector_original = Q2(Q1Values)
    # Q3("UofT.jpg", 512, 512, 90, 2)
    # Q3("UofT_1.jpg", 512, 512, 0, 2)
    # change1 = Q1("UofT_rotated.png", 30, 5)
    # feature_vector_change2 = Q2(change1)
    # print(len(change1[0]))
    # feature_vector_change1 = Q2(Q1("UofT_1_rotated.png", 30, 5))
    # Q4("UofT_1_rotated.png", "UofT_rotated.png", feature_vector_change1, feature_vector_change2, ((512, 512), 90, 1), 5)
    # Q7()
    Q8('./my_apartment/image_1.png', './my_apartment/image_2.png', 100)

