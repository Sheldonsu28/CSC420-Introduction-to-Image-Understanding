import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D



# Q1
def compute_point_cloud(num):
    path = "./Part1/" + str(num) + "/"
    depth_map = cv.imread(path + "depthImage.png")
    rgb_image = cv.imread(path + "rgbImage.jpg")
    extrinsic = np.loadtxt(path + "extrinsic.txt")
    intrinsic = np.loadtxt(path + "intrinsics.txt")
    result_vector = np.empty((depth_map.shape[0] * depth_map.shape[1], 6))
    iteration = 0
    for y in range(0, depth_map.shape[0]):
        for x in range(0, depth_map.shape[1]):
            img_coord = depth_map[y, x] * np.array([x, y, 1])

            cam_coord = np.linalg.inv(intrinsic) @ img_coord

            world_coord = np.linalg.inv(extrinsic[:, :-1]) @ (cam_coord - extrinsic[:, -1])

            result_vector[iteration, :3] = world_coord
            result_vector[iteration, 3:] = rgb_image[y, x]

            iteration += 1

    np.savetxt(path + "pointCloud.txt", result_vector)


# Q2
def get_rotation_matrix(axis, rad):
    if axis == 'y':
        return np.array([[1, 0, 0],
                         [0, np.cos(rad), -np.sin(rad)],
                         [0, np.sin(rad), np.cos(rad)]])

    elif axis == 'x':
        return np.array([[np.cos(rad), 0, np.sin(rad)],
                         [0, 1, 0],
                         [-np.sin(rad), 0, np.cos(rad)]])

    return np.array([[np.cos(rad), -np.sin(rad), 0],
                     [np.sin(rad), np.cos(rad), 0],
                     [0, 0, 1]])


def rotate_coordinate(omega_t, axis, num):
    path = "./Part1/" + str(num) + "/"
    points_array = np.loadtxt(path + "pointCloud.txt")
    rotation_matix = get_rotation_matrix(axis, omega_t)
    rgb_image = cv.imread(path + "rgbImage.jpg")
    extrinsic = np.loadtxt(path + "extrinsic.txt")
    intrinsic = np.loadtxt(path + "intrinsics.txt")

    rotated_img = np.zeros(rgb_image.shape)
    rotated_depth = np.zeros((rotated_img.shape[0], rotated_img.shape[1]))
    y_max, x_max, _ = rotated_img.shape

    for i in range(points_array.shape[0]):
        camera_coordinate = extrinsic @ np.array([points_array[i, 0], points_array[i, 1], points_array[i, 2], 1])

        rotated_camera_coordinate = rotation_matix @ camera_coordinate
        img_coord = intrinsic @ rotated_camera_coordinate

        w = img_coord[-1]
        img_coord /= w if w != 0 else 1

        y, x = int(np.rint(img_coord[1])), int(np.rint(img_coord[0]))

        if 0 <= y < y_max and 0 <= x < x_max:
            rotated_img[y, x, :] = points_array[i, 3:]
            rotated_depth[y, x] = w

    return rotated_depth, rotated_img


# Q3
def create_img_sequence():
    axis = ["x", "y", 'z']
    for i in range(1, 4):
        angles = np.linspace(0, np.pi/4, 16)
        img_array = []
        img_depth_array = []
        path = "./Part1/" + str(i) + "/"
        r_axis = axis[i - 1]
        for j in range(len(angles)):
            omega = angles[j]
            rotated_depth, rotated_img = rotate_coordinate(omega, r_axis, i)
            img_array.append(rotated_img)
            img_depth_array.append(rotated_depth)
            if j % 4 == 0:
                cv.imwrite(path + "rotated_depth_Omega" + str(omega) + '_axis_' + r_axis + "_.png", rotated_depth)
                cv.imwrite(path + "rotated_img_Omega" + str(omega) + '_axis_' + r_axis + "_.png", rotated_img)

        video = cv.VideoWriter(path + '_axis_' + r_axis + ".avi", cv.VideoWriter_fourcc(*'DIVX'), 15, (640, 480))
        for img in img_array:
            video.write(np.uint8(img))
        video.release()

# Q4
def fill_in_A(points):
    n = 0
    for obj in points:
        for match in obj:
            n += 1
    A = np.empty((n, 9))
    i = 0

    for obj in points:
        for match in obj:
            xl, yl = match[0][0], match[0][1]
            xr, yr = match[1][0], match[1][1]
            A[i, :] = [xr * xl, xr * yl, xr, yr * xl, yr * yl, yr, xl, yl, 1]
            i += 1
    return A


def Eight_point(points):
    A = fill_in_A(points)
    U, D, VT = np.linalg.svd(A)
    F = VT[8].reshape(3, 3)
    U1, D1, VT1 = np.linalg.svd(F)
    D1[-1] = 0
    F_prime = (U1 * D1) @ VT1
    return F_prime


# Q5
def Q5(F):
    U, D, VT = np.linalg.svd(F)
    el = VT[-1]
    er = U[:, -1]
    return el, er


def Q6(points, F, img_name, name, i):
    M = np.array([[-35e-3 / 26.111e-6, 0, 450],
                  [0, -35e-3 / 26.111e-6, 300],
                  [0, 0, 1]])
    points1 = []
    points2 = []
    for obj_points in points:
        for point_pair in obj_points:
            points1.append(point_pair[0])
            points2.append(point_pair[1])
    # f_prime = cv.findFundamentalMat(np.array(points1), np.array(points2), cv.FM_8POINT)
    #
    # F = f_prime[0]
    E = M.T @ F @ M
    R1, R2, t = cv.decomposeEssentialMat(E)
    RS = [R1, R2]
    t = t.reshape(3)
    R = RS[i]

    A = np.empty((3, 3))

    real_points = []
    point = []
    for obj_points in points:
        for point_pair in obj_points:
            pl = point_pair[0]
            pr = point_pair[1]
            RT = R.T
            A[:, 0] = pl
            A[:, 1] = RT @ pr
            A[:, 2] = np.cross(pl, RT @ pr)
            result_vector = np.linalg.solve(A, t)
            a, b, c = result_vector[0], result_vector[1], result_vector[2]
            P_prime = ((t + b * RT @ pr) + a * pl)/2
            real_points.append(P_prime)
            point.append(pl)

    img = cv.imread(img_name)
    real_points = np.array(real_points)
    point = np.array(point)
    constant = np.max(real_points[:, 2]) - np.min(real_points[:, 2])
    min_p = np.min(real_points[:, 2])

    for i in range(real_points.shape[0]):
        cv.circle(img, (point[i, 0], point[i, 1]), 3, ((real_points[i, 2] - min_p)*255/constant, 0, 255 - (real_points[i, 2] - min_p)*255/constant))

        cv.putText(img, "{:.4f}".format(float(real_points[i, 2])), (point[i, 0], point[i, 1]), cv.FONT_HERSHEY_SIMPLEX,
                   0.4, ((real_points[i, 2] - min_p)*255/constant, 0, 255 - (real_points[i, 2] - min_p)*255/constant))

    cv.imwrite(name, img)


if __name__ == "__main__":
    for num in range(1, 4):
        compute_point_cloud(num)
    create_img_sequence()
    # img = mpimg.imread("./Part2/second_pair/p22.jpg")
    # plt.imshow(img)
    # result = plt.ginput(4)
    # print(result)

    # Q4 result
    test_n_8 = [[
        [np.array([349, 154, 1]), np.array([380, 172, 1])],
        [np.array([726, 155, 1]), np.array([741, 177, 1])],
        [np.array([735, 380, 1]), np.array([747, 394, 1])],
        [np.array([352, 390, 1]), np.array([383, 404, 1])]
    ], [
        [np.array([234, 390, 1]), np.array([277, 406, 1])],
        [np.array([264, 382, 1]), np.array([300, 396, 1])],
        [np.array([328, 385, 1]), np.array([364, 399, 1])],
        [np.array([301, 393, 1]), np.array([345, 408, 1])]

    ]]
    print(Eight_point(test_n_8))

    test_n_8_add_1 = [[
        [np.array([350, 154, 1]), np.array([381, 172, 1])],
        [np.array([726, 155, 1]), np.array([741, 178, 1])],
        [np.array([735, 380, 1]), np.array([747, 394, 1])],
        [np.array([352, 390, 1]), np.array([383, 404, 1])]
    ], [
        [np.array([234, 390, 1]), np.array([277, 406, 1])],
        [np.array([264, 382, 1]), np.array([300, 396, 1])],
        [np.array([328, 385, 1]), np.array([364, 399, 1])],
        [np.array([301, 393, 1]), np.array([345, 408, 1])]

    ]]
    print(Eight_point(test_n_8_add_1))

    test_n_12 = [[
        [np.array([350, 154, 1]), np.array([381, 172, 1])],
        [np.array([726, 155, 1]), np.array([741, 178, 1])],
        [np.array([735, 380, 1]), np.array([747, 394, 1])],
        [np.array([352, 390, 1]), np.array([383, 404, 1])]
    ], [
        [np.array([234, 390, 1]), np.array([277, 406, 1])],
        [np.array([264, 382, 1]), np.array([300, 396, 1])],
        [np.array([328, 385, 1]), np.array([364, 399, 1])],
        [np.array([301, 393, 1]), np.array([345, 408, 1])]
    ], [
        [np.array([264, 522, 1]), np.array([325, 535, 1])],
        [np.array([283, 497, 1]), np.array([331, 511, 1])],
        [np.array([342, 497, 1]), np.array([390, 511, 1])],
        [np.array([328, 522, 1]), np.array([390, 535, 1])]
    ]]
    print(Eight_point(test_n_12))

    #Q5

    part2_pair1_points = [[
        [np.array([347, 153, 1]), np.array([387, 165, 1])],
        [np.array([726, 156, 1]), np.array([740, 177, 1])],
        [np.array([735, 380, 1]), np.array([745, 394, 1])],
        [np.array([352, 390, 1]), np.array([383, 403, 1])]
    ], [
        [np.array([234, 391, 1]), np.array([277, 408, 1])],
        [np.array([264, 382, 1]), np.array([302, 396, 1])],
        [np.array([328, 385, 1]), np.array([364, 399, 1])],
        [np.array([300, 393, 1]), np.array([348, 408, 1])]
    ], [
        [np.array([264, 522, 1]), np.array([325, 535, 1])],
        [np.array([283, 497, 1]), np.array([331, 511, 1])],
        [np.array([342, 497, 1]), np.array([390, 511, 1])],
        [np.array([328, 522, 1]), np.array([390, 535, 1])]
    ], [
        [np.array([180, 543, 1]), np.array([251, 559, 1])],
        [np.array([217, 541, 1]), np.array([286, 555, 1])],
        [np.array([191, 479, 1]), np.array([258, 496, 1])],
        [np.array([201, 478, 1]), np.array([268, 494, 1])]
    ], [
        [np.array([429, 476, 1]), np.array([470, 491, 1])],
        [np.array([771, 465, 1]), np.array([796, 478, 1])],
        [np.array([432, 522, 1]), np.array([484, 534, 1])],
        [np.array([808, 510, 1]), np.array([839, 523, 1])]

    ]]

    part2_pair1_points_delete = [[
        [np.array([347, 153, 1]), np.array([387, 165, 1])],
        [np.array([726, 156, 1]), np.array([740, 177, 1])],
        [np.array([735, 380, 1]), np.array([745, 394, 1])],
        [np.array([352, 390, 1]), np.array([383, 403, 1])]
    ], [
        [np.array([234, 391, 1]), np.array([277, 408, 1])],
        [np.array([264, 382, 1]), np.array([302, 396, 1])],
        [np.array([328, 385, 1]), np.array([364, 399, 1])],
        [np.array([300, 393, 1]), np.array([348, 408, 1])]
    ], [
        [np.array([264, 522, 1]), np.array([325, 535, 1])],
        [np.array([283, 497, 1]), np.array([331, 511, 1])],
        [np.array([342, 497, 1]), np.array([390, 511, 1])],
        [np.array([328, 522, 1]), np.array([390, 535, 1])]
    ], [
        [np.array([429, 476, 1]), np.array([470, 491, 1])],
        [np.array([771, 465, 1]), np.array([796, 478, 1])],
        [np.array([432, 522, 1]), np.array([484, 534, 1])],
        [np.array([808, 510, 1]), np.array([839, 523, 1])]

    ]]

    part2_pair2_points = [[
        [np.array([385, 93, 1]), np.array([412, 104, 1])],
        [np.array([771, 75, 1]), np.array([830, 90, 1])],
        [np.array([394, 329, 1]), np.array([419, 343, 1])],
        [np.array([769, 292, 1]), np.array([826, 320, 1])]
    ], [
        [np.array([324, 363, 1]), np.array([332, 373, 1])],
        [np.array([348, 338, 1]), np.array([371, 357, 1])],
        [np.array([422, 345, 1]), np.array([442, 362, 1])],
        [np.array([397, 371, 1]), np.array([406, 389, 1])]
    ], [
        [np.array([694, 395, 1]), np.array([737, 426, 1])],
        [np.array([458, 370, 1]), np.array([530, 448, 1])],
        [np.array([638, 346, 1]), np.array([691, 370, 1])]
    ], [
        [np.array([740, 468, 1]), np.array([757, 500, 1])],
        [np.array([502, 492, 1]), np.array([499, 517, 1])],
        [np.array([514, 537, 1]), np.array([500, 568, 1])],
        [np.array([769, 515, 1]), np.array([773, 560, 1])]
    ], [
        [np.array([398, 507, 1]), np.array([392, 527, 1])],
        [np.array([329, 510, 1]), np.array([325, 526, 1])],
        [np.array([330, 569, 1]), np.array([305, 593, 1])],
        [np.array([406, 570, 1]), np.array([378, 592, 1])]

    ]]

    part2_pair2_points_delete = [[
        [np.array([385, 93, 1]), np.array([412, 104, 1])],
        [np.array([771, 75, 1]), np.array([830, 90, 1])],
        [np.array([394, 329, 1]), np.array([419, 343, 1])],
        [np.array([769, 292, 1]), np.array([826, 320, 1])]
    ], [
        [np.array([320, 363, 1]), np.array([332, 376, 1])],
        [np.array([349, 338, 1]), np.array([371, 350, 1])],
        [np.array([422, 345, 1]), np.array([442, 362, 1])],
        [np.array([397, 371, 1]), np.array([406, 389, 1])]
    ], [
        [np.array([346, 426, 1]), np.array([349, 440, 1])],
        [np.array([373, 498, 1]), np.array([375, 517, 1])],
        [np.array([443, 402, 1]), np.array([451, 419, 1])],
        [np.array([453, 479, 1]), np.array([462, 497, 1])]

    ]]

    print(Q5(Eight_point(part2_pair1_points)))
    print(Q5(Eight_point(part2_pair1_points_delete)))

    print(Q5(Eight_point(part2_pair2_points)))
    print(Q5(Eight_point(part2_pair1_points_delete)))


    F1 = Eight_point(part2_pair1_points)


    Q6(part2_pair1_points, F1, "./Part2/first_pair/p11.jpg", "Q6_pair1.png", 0)

    F2 = Eight_point(part2_pair2_points)

    Q6(part2_pair2_points, F2, "./Part2/second_pair/p21.jpg", "Q6_pair2.png", 0)
