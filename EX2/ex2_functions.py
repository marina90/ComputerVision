import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import random

DEBUG = True


def compute_homography_naive(match_p_src, match_p_dst):
    num_of_points = np.shape(match_p_src)[1]
    match_p_src = np.concatenate((match_p_src.copy(), np.ones([1, np.shape(match_p_src)[1]])), axis=0)
    match_p_dst = np.concatenate((match_p_dst.copy(), np.ones([1, np.shape(match_p_src)[1]])), axis=0)

    A_matrix = np.empty((num_of_points * 2, 9))
    for ii in range(num_of_points):
        # Good explanation
        # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
        cur_src_point = match_p_src[:, ii]
        cur_dst_point = match_p_dst[:, ii]

        cur_u_vec = np.append(np.append(cur_src_point, np.zeros(3)), -cur_dst_point[0] * cur_src_point)
        cur_v_vec = np.append(np.append(np.zeros(3), cur_src_point), -cur_dst_point[1] * cur_src_point)

        A_matrix[2 * ii, :] = cur_u_vec
        A_matrix[2 * ii + 1, :] = cur_v_vec

    U, S, V = np.linalg.svd(A_matrix)

    V = V.T

    H_vec = V[:, -1]
    H = np.reshape(H_vec, (3, 3))

    return H


def test_homography(H, mp_src, mp_dst, max_err):
    # README notes
    ## Inputs:
    ### mp_src = A variable containing 2 rows and N columns, where the i column represents coordinates of match point i in the src image
    ### mp_dst = A variable containing 2 rows and N columns, where the i column represents coordinates of match point i in the dst image
    ### max_err = A scalar that represents the maximum distance (in pixels) between the mapped src point to its corresponding dst ...
    ### ... point, in order to be considered as valid inlier
    ## Outputs:
    ### fit_percent = The probability (between 0 and 1) validly mapped src points (inliers)
    ### dist_mse = Mean square error of the distances between validly mapped src points, to their corresponding dst points (only for inliers).

    ## Create a (u,v,1) matrix for all src points
    # mp_src_3d = np.append(mp_src, np.ones_like(mp_src[0:1]),axis=0)
    mp_src_t = (mp_src.transpose()).reshape(-1, 1, 2)

    # Calculate and normalize dst from src & H
    mp_dst_estimated = cv2.perspectiveTransform(np.float32(mp_src_t.reshape(-1, 1, 2)), np.float32(H))
    # mp_dst_estimated = np.matmul(H, mp_src_3d)
    # mp_dst_est_norm = np.divide(mp_dst_estimated, mp_dst_estimated[2])
    mp_dst_t = np.transpose(mp_dst)

    # calculate distance between estimation and destination
    mp_dst_estimated_2D = mp_dst_estimated[:, 0]
    dst_diff = mp_dst_estimated[:, 0] - mp_dst_t
    distance = np.sqrt(np.power(dst_diff[:, 0], 2) + np.power(dst_diff[:, 1], 2))

    # pass TH for inlier definition
    inlier_indices = distance < max_err
    inlier_points_dist = distance[inlier_indices]

    # calculate percentage of data fitting model
    fit_percent = len(inlier_points_dist) / len(distance)

    # Calculate MSE for inliers
    dist_mse = np.mean(np.square(inlier_points_dist))

    return fit_percent, dist_mse


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    # README notes
    ## Inputs:
    ### mp_src = A variable containing 2 rows and N columns, where the i column represents coordinates of match point i in the src image
    ### mp_dst = A variable containing 2 rows and N columns, where the i column represents coordinates of match point i in the dst image
    ### inliers_percent = The expected probability (between 0 and 1) of correct match points from the entire list of match points
    ### max_err = A scalar that represents the maximum distance (in pixels) between the mapped src point to its corresponding dst point, ...
    ### ...in order to be considered as valid inlier
    ## Outputs:
    ### H = Projective transformation matrix from src to dst

    # Define parameters
    ## TODO: calculate k
    n = 4  # number of points for homography
    k = 30  # number of iterations
    indices = range(0, len(mp_src[0]) - 1)

    # Run RANSAC
    for i in range(0, k - 1):

        ## sample n indices
        indices_i = np.asarray(random.sample(indices, n))
        batch_points_src = np.asarray([mp_src[0, indices_i], mp_src[1, indices_i]])
        batch_points_dst = np.asarray([mp_dst[0, indices_i], mp_dst[1, indices_i]])

        ## calculate model
        H_i = compute_homography_naive(batch_points_src, batch_points_dst)

        ## calculate model fit
        fit_percent_i, dist_mse_i = test_homography(H_i, mp_src, mp_dst, max_err)
        if (fit_percent_i >= inliers_percent):
            ## calcaulte the model using all inlier points
            ## TODO: consider writing it as a seperate function
            mp_src_t = (mp_src.transpose()).reshape(-1, 1, 2)
            # Calculate and normalize dst from src & H
            mp_dst_estimated = cv2.perspectiveTransform(np.float32(mp_src_t.reshape(-1, 1, 2)), np.float32(H_i))
            mp_dst_t = np.transpose(mp_dst)
            # calculate distance between estimation and destination
            dst_diff = mp_dst_estimated[:, 0] - mp_dst_t
            distance = np.sqrt(np.power(dst_diff[:, 0], 2) + np.power(dst_diff[:, 1], 2))
            # pass TH for inlier definition
            inlier_indices = distance < max_err
            H_i_all_inliers = compute_homography_naive(mp_src[:, inlier_indices], mp_dst[:, inlier_indices])

            return H_i_all_inliers

    print('RANSAC has failed to find a model which comply to inliers percent of ')
    print(inliers_percent)
    print('and max error of ')
    print(max_err)
    return False


def panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err):
    # TODO: add here ransac homography when Yonatan finish his part
    H = compute_homography_naive(match_p_src, match_p_dst)
    transformed_img_src = cv2.warpPerspective(img_src, H, np.shape(img_src)[0:2])

    upper_left_point = find_upper_left_point(match_p_src)

    if True:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img_src)
        plt.title('img_src')
        plt.scatter(upper_left_point[0], upper_left_point[1], marker='x', color='r')
        plt.subplot(2, 2, 2)
        plt.imshow(transformed_img_src)
        plt.title('transformed_img_src')
        plt.scatter(match_p_dst[0, :], match_p_dst[1, :], marker='x', color='r')
        plt.show(block=False)

    pad_x = round(2 * np.shape(img_dst)[0])
    pad_y = np.shape(img_dst)[1]

    transformed_img_dst = cv2.warpPerspective(img_dst, np.linalg.inv(H),
                                              (np.shape(img_src)[0] + pad_x, np.shape(img_src)[1] + pad_y))

    if DEBUG:
        plt.subplot(2, 2, 3)
        plt.imshow(img_dst)
        plt.title('img_dst')
        plt.subplot(2, 2, 4)
        plt.imshow(transformed_img_dst)
        plt.title('transformed_img_dst')
        plt.show(block=False)

    output_image = np.uint8(np.zeros(np.shape(transformed_img_dst)))
    output_image[0:np.shape(transformed_img_dst)[0], 0:np.shape(transformed_img_dst)[1],
    :] = output_image + transformed_img_dst
    output_image[0:np.shape(img_src)[0], 0:np.shape(img_src)[1], :] = img_src

    if DEBUG:
        plt.figure()
        plt.imshow(output_image)
        plt.show(block=False)

    return


def find_upper_left_point(match_p_src):
    dist_from_0_0 = np.linalg.norm(match_p_src.T, axis=1)
    arg_min = np.argmin(dist_from_0_0)

    return match_p_src[:, arg_min]


def panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err):
    H = compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
    transformed_img_src = cv2.warpPerspective(img_src, H, np.shape(img_src)[0:2])

    if DEBUG:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img_src)
        plt.title('img_src')
        plt.subplot(2, 2, 2)
        plt.imshow(transformed_img_src)
        plt.title('transformed_img_src')
        plt.scatter(match_p_dst[0, :], match_p_dst[1, :], marker='x', color='r')
        plt.show(block=False)

    dst_shape = np.shape(img_dst)[0:2]
    dst_img_4_points = np.array([[0, 0], [0, dst_shape[1]], [dst_shape[0], 0], [dst_shape[0], dst_shape[1]]])
    dst_img_4_points_transformed = cv2.perspectiveTransform(np.float32(dst_img_4_points).reshape(-1, 1, 2),
                                                            np.linalg.inv(H)).reshape(-1, 2)

    if DEBUG:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img_src)
        plt.title('img_src')
        plt.scatter(dst_img_4_points_transformed[:, 0], dst_img_4_points_transformed[:, 1], marker='x', color='r')

    pad_x = int(np.ceil(np.max(dst_img_4_points_transformed[:, 0])))
    pad_y = int(np.ceil(np.max(dst_img_4_points_transformed[:, 1])))

    transformed_img_dst = cv2.warpPerspective(img_dst, np.linalg.inv(H),
                                              (pad_x, pad_y))

    if DEBUG:
        plt.subplot(2, 2, 3)
        plt.imshow(img_dst)
        plt.title('img_dst')
        plt.subplot(2, 2, 4)
        plt.imshow(transformed_img_dst)
        plt.title('transformed_img_dst')
        plt.show(block=False)

    output_image = np.uint8(np.zeros(np.shape(transformed_img_dst)))
    output_image[0:np.shape(transformed_img_dst)[0], 0:np.shape(transformed_img_dst)[1],
    :] = output_image + transformed_img_dst
    output_image[0:np.shape(img_src)[0], 0:np.shape(img_src)[1], :] = img_src

    if DEBUG:
        plt.figure()
        plt.imshow(output_image)
        plt.show(block=False)

    return
