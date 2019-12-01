## Import needed packages

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import random

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

    # Create a (u,v,1) matrix for all src points
    mp_src_3d = np.append(mp_src, np.ones_like(mp_src[0:1]),axis=0)

    # Calculate and normalized dst from src & H
    mp_dst_estimated = np.matmul(H, mp_src_3d)
    mp_dst_est_norm = np.divide(mp_dst_estimated, mp_dst_estimated[2])

    # calculate distance between estimation and destination
    dst_diff = mp_dst_est_norm[0:2,:] - mp_dst
    distance = np.sqrt(np.power(dst_diff[0],2) + np.power(dst_diff[1],2))

    # pass TH for inlier definition
    inlier_indices = distance < max_err
    inlier_points = distance[inlier_indices]

    # calculate percentage of data fitting model
    fit_percent = len(inlier_points)/len(distance)

    # Calculate MSE for inliers
    dist_mse = np.mean(np.square(inlier_points))


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
    n = 4   # number of points for homography
    k = 30  # number of iterations
    indices = np.indices([len(mp_src[0])])

    # Run RANSAC
    for i in range (0,k-1):

        ## for debug
        if(1):
            print ('RANSAC iteration ' + i + 'out of ' k-1)

        ## sample n indices
        indices_i = random.sample(indices, n)
        batch_points_src = mp_src[indices_i,:]
        batch_points_dst = mp_dst[indices_i,:]

        ## calculate model
        H_i = compute_homography_naive(batch_points_src, batch_points_dst)

        ## calculate model fit
        fit_percent_i, dist_mse_i = test_homography(H_i, mp_src, mp_dst,max_err)
        if (fit_percent_i >= inliers_percent):
            # H_i_all_inliers = compute_homography_naive(batch_points_src, batch_points_dst)
            return H_i_all_inliers

    print('RANSAC has failed to find a model which comply to inliers percent of ' + inliers_percent + 'and max error of ' + max_err)

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

    pad_x = round(2*np.shape(img_dst)[0])
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
    output_image[0:np.shape(transformed_img_dst)[0], 0:np.shape(transformed_img_dst)[1], :] = output_image + transformed_img_dst
    output_image[0:np.shape(img_src)[0], 0:np.shape(img_src)[1], :] = img_src

    if DEBUG:
        plt.figure()
        plt.imshow(output_image)
        plt.show(block=False)

    return