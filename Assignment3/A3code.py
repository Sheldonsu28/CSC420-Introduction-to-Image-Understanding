import numpy as np
import scipy as sp
import cv2 as cv
from scipy import signal as sps
from scipy import optimize as spo
from matplotlib import pyplot as plt
from Assignment3 import A2Q7 as q7
from scipy.io import loadmat


# Code for Q1
def Q1(imgName, threshlod, kernel=3):
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
            weighted_magnitude = gradient_magnitude * kernel
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
        flatten_orientation = np.ndarray.flatten(v[0]) * 180 / np.pi
        flatten_weighted_magnitude = np.ndarray.flatten(v[1])
        location_vector = v[-1]
        counts = np.zeros(36)
        for i in range(0, 36):
            counts[i] = sum(
                flatten_weighted_magnitude[(flatten_orientation >= i * 10) & (flatten_orientation < 10 * (i + 1))])
        peak_index = np.argmax(counts)
        for i in range(0, 36):
            location_vector.append(counts[int((peak_index + i) % 36)])
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
    vector1 = vector1 / np.sqrt(np.sum(vector1 ** 2))
    vector2 = vector2 / np.sqrt(np.sum(vector2 ** 2))
    BC = np.sum(np.sqrt(vector1 * vector2))
    return BC


def Q4(originalImage, rotatedImage, original_vectors, rotated_vectors, rotation_param, window_size):
    orig_image = cv.imread(originalImage)
    rot_image = cv.imread(rotatedImage)
    output = np.empty((orig_image.shape[0], 2 * orig_image.shape[1], orig_image.shape[-1]))
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
    for m in matched_vectors:
        value = m
        src = value[1]
        target = value[0]
        point1 = (int(shape[1] + src[1] * 2 ** src[-1]), int(src[0] * 2 ** src[-1]))
        point2 = (int(target[1] * 2 ** target[-1]), int(target[0] * 2 ** target[-1]))
        cv.line(output, point1, point2, (255, 0, 0))
    cv.imwrite(rotatedImage[0:-4] + "_match.png", output)


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


def Q8(img1, img2, feature_num, write=False):
    orig_img1 = cv.imread(img1) if isinstance(img1, str) else img1
    orig_img2 = cv.imread(img2) if isinstance(img2, str) else img2


    sift = cv.xfeatures2d.SIFT_create(nfeatures=feature_num)

    kps1, descriptor1 = sift.detectAndCompute(orig_img1, None)
    kps2, descriptor2 = sift.detectAndCompute(orig_img2, None)

    orig_img1 = cv.drawKeypoints(orig_img1, kps1, None)
    orig_img2 = cv.drawKeypoints(orig_img2, kps2, None)
    if write == True:
        cv.imwrite(img1[:-3] + 'sift_detect.jpg', orig_img1)
        cv.imwrite(img2[:-3] + 'sift_detect.jpg', orig_img2)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    match_1 = []
    r_kps1 = []
    r_kps2 = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            r_kps1.append(kps1[m.queryIdx])
            r_kps2.append(kps2[m.trainIdx])
            match_1.append([m])
    if write:
        output = cv.drawMatchesKnn(orig_img1, kps1, orig_img2, kps2, match_1, None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite("Q8_match.png", output)
    return r_kps1, r_kps2


def RNG(low, high, size):
    r_list = np.empty(size, dtype=int)
    r_list.fill(-1)
    count = 0
    while count < size:
        a = int(np.random.randint(low, high))
        while np.any(r_list == a):
            a = int(np.random.randint(low, high))
        r_list[count] = a
        count += 1
    return r_list


def Homographic_matrix_generator():
    matrix = np.zeros((9, 9))
    matrix[:, 2] = [-1, 0, -1, 0, -1, 0, -1, 0, 0]
    matrix[:, 5] = [0, -1, 0, -1, 0, -1, 0, -1, 0]
    return matrix


def Homographic_matrix_filler(matrix, kps1, kps2):
    x1, x2, x3, x4 = kps1[0, 0], kps1[0, 1], kps1[0, 2], kps1[0, 3]
    y1, y2, y3, y4 = kps1[1, 0], kps1[1, 1], kps1[1, 2], kps1[1, 3]
    xp1, xp2, xp3, xp4 = kps2[0, 0], kps2[0, 1], kps2[0, 2], kps2[0, 3]
    yp1, yp2, yp3, yp4 = kps2[1, 0], kps2[1, 1], kps2[1, 2], kps2[1, 3]

    matrix[:, 0] = [-x1, 0, -x2, 0, -x3, 0, -x4, 0, 0]
    matrix[:, 1] = [-y1, 0, -y2, 0, -y3, 0, -y4, 0, 0]

    matrix[:, 3] = [0, -x1, 0, -x2, 0, -x3, 0, -x4, 0]
    matrix[:, 4] = [0, -y1, 0, -y2, 0, -y3, 0, -y4, 0]

    matrix[:, 6] = [x1 * xp1, x1 * yp1, x2 * xp2, x2 * yp2, x3 * xp3, x3 * yp3, x4 * xp4, x4 * yp4, 0]
    matrix[:, 7] = [y1 * xp1, y1 * yp1, y2 * xp2, y2 * yp2, y3 * xp3, y3 * yp3, y4 * xp4, y4 * yp4, 0]
    matrix[:, 8] = [xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4, 1]
    return matrix


def Q9(img1, img2, kps1, kps2, threshold, limit=1000, write=False):
    image1 = cv.imread(img1) if isinstance(img1, str) else img1
    image2 = cv.imread(img2) if isinstance(img2, str) else img2
    shape = image1.shape
    kps1_len = len(kps1)
    kps2_len = len(kps2)

    kps1_coord = np.empty((2, kps1_len))
    kps2_coord = np.empty((2, kps2_len))
    for i in range(kps1_len):
        kps1_coord[:, i] = kps1[i].pt

    for i in range(kps2_len):
        kps2_coord[:, i] = kps2[i].pt

    best = 0
    H = None
    matched_point = []
    kps1_len = kps1_coord.shape[1]
    matrix = Homographic_matrix_generator()
    iteration = 0
    b = np.zeros(9)
    b[-1] = 1
    while best < threshold and iteration < limit:
        temp = []
        random = RNG(0, kps1_len, 4)
        m1 = kps1_coord[:, random]
        m2 = kps2_coord[:, random]
        A = Homographic_matrix_filler(matrix[:, :], m1, m2)
        result = spo.lsq_linear(A, b).x.reshape((3,3))
        c_vector = np.ones(3)
        count = 0
        for i in range(kps1_len):
            c_vector[:-1] = kps1_coord[:, i]
            transformed = result @ c_vector
            new_coord = transformed[:2] / transformed[-1]
            if np.allclose(new_coord, kps2_coord[:, i], atol=1):
                temp.append((kps1_coord[:, i], kps2_coord[:, i]))
                count += 1
        if count > best:
            H = result[:, :]
            best = count
            matched_point = temp[:]
        iteration += 1
    if write:
        output = np.empty((shape[0], 2 * shape[1], shape[2]))
        output[:, :shape[1], :] = image1
        output[:, shape[1]:, :] = image2

        for match in matched_point:
            point1 = (int(match[0][0]), int(match[0][1]))
            point2 = (int(match[1][0]) + shape[1], int(match[1][1]))
            cv.line(output, point1, point2, (255, 0, 0))
        cv.imwrite('Q9_match.png', output)
    if iteration == threshold:
        print('Threshold reach, terminating')
    return H


def Q10(img1, img2, inv=False):

    kps1, kps2 = Q8(img2, img1, 3000)
    H = Q9(img1, img2, kps1, kps2, 900)
    img1 = cv.imread(img1) if isinstance(img1, str) else img1
    img2 = cv.imread(img2) if isinstance(img2, str) else img2

    T_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    max_value = []
    for y in range(img2.shape[0]):
        last_point = H @ np.array([[img2.shape[1] - 1], [y], [1]])
        first_point = last_point[0] / last_point[-1]
        last_point1 = H @ np.array([[0], [y], [1]])
        first_point1 = last_point1[0] / last_point1[-1]
        max_value.append(int(first_point))
        max_value.append(int(first_point1))

    max_transform = max(max_value) + 1
    min_transform = min(max_value) + 1
    furthest_x = max_transform if max_transform > img1.shape[1] else img1.shape[1]
    closest_x = min_transform if min_transform < 0 else 0
    T_matrix[0, 2] = abs(closest_x)
    if inv:
        result = cv.warpPerspective(img2, H @ T_matrix, (img1.shape[1] + abs(closest_x), img1.shape[0]))

        bound = int(0.01 * img1.shape[1])

        blend2 = np.where(result[:, abs(closest_x):abs(closest_x)+bound, :] < 0.1 * img1[:, :bound, :], img1[:, :bound, :],
                          result[:, abs(closest_x):abs(closest_x)+bound, :])
        result[:, abs(closest_x):abs(closest_x) + bound, :] = blend2

        blend1 = np.where(img1[:, :bound, :] < 0.1 * result[:, abs(closest_x):abs(closest_x) + bound, :],
                          result[:, abs(closest_x):abs(closest_x) + bound, :],
                          img1[:, :bound, :])
        img1[:, :bound] = blend1

        result[:img1.shape[0], abs(closest_x):] = img1

    else:
        result = cv.warpPerspective(img2, H, (furthest_x, img1.shape[0]))
        bound = int(0.95*img1.shape[1])
        max_v = img1.shape[1]

        blend2 = np.where(result[:, bound:max_v, :] < 0.1 * img1[:, bound:, :], img1[:, bound:, :],
                          result[:, bound:max_v, :])
        result[:, bound:max_v, :] = blend2

        blend1 = np.where(img1[:, bound:, :] < 0.1 * result[:, bound:max_v, :], result[:, bound:max_v, :],
                          img1[:, bound:, :])
        img1[:, bound:] = blend1
        result[:img1.shape[0], :img1.shape[1]] = img1

    while np.sum(result[:, 0, :]) < 10*255:
        result = result[:, 1:]
    while np.sum(result[:, -1, :]) < 10*255:
        result = result[:, :-1, :]
    return result


if __name__ == "__main__":
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
    # kps1, kps2 = Q8('./my_apartment/image_1.png', './my_apartment/image_2.png', 100)
    # Q9('./my_apartment/image_1.png', './my_apartment/image_2.png', kps1, kps2, 20)
    img_list = ['./my_apartment/image_1.png', './my_apartment/image_2.png', './my_apartment/image_3.png',
                './my_apartment/image_4.png', './my_apartment/image_5.png', './my_apartment/image_6.png',
                './my_apartment/image_7.png', './my_apartment/image_8.png', './my_apartment/image_9.png',
                './my_apartment/image_10.png']
    result_56 = Q10(img_list[5], img_list[6])
    result_87 = Q10(img_list[8], img_list[7])
    result_89 = Q10(img_list[8], img_list[9])
    result_789 = Q10(result_87, result_89)
    result_67 = Q10(img_list[6], img_list[7])
    result_567 = Q10(result_56, result_67)
    result_56789 = Q10(result_567, result_789)
    result_43 = Q10(img_list[4], img_list[3], inv=True)
    result_32 = Q10(img_list[3], img_list[2], inv=True)
    result_21 = Q10(img_list[2], img_list[1], inv=True)
    result_10 = Q10(img_list[1], img_list[0], inv=True)
    result_210 = Q10(result_21, result_10, inv=True)
    result_432 = Q10(result_43, result_32, inv=True)
    result_43210 = Q10(result_432, result_210, inv=True)
    final_result = Q10(result_43210, result_56789)
    cv.imwrite("Q10_final.png", final_result)
