## Import needed packages

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from ex2_functions import *
import matplotlib.image as mpimg

## Load data
src = mpimg.imread('Input_Files/src.jpg')
dst = mpimg.imread('Input_Files/dst.jpg')
matches = sio.loadmat('Input_Files/matches.mat')
match_perfect = sio.loadmat('Input_Files/matches_perfect.mat')

# Display images + perfect matching points
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(src)
plt.title('src with perfect mp')
plt.scatter(x=match_perfect['match_p_src'][0], y=match_perfect['match_p_src'][1], c='r', s=10)

plt.subplot(2, 2, 2)
plt.imshow(dst)
plt.title('dst with perfect mp')
plt.scatter(x=match_perfect['match_p_dst'][0], y=match_perfect['match_p_dst'][1], c='r', s=10)

plt.subplot(2, 2, 3)
plt.imshow(src)
plt.title('src with outliers mp')
plt.scatter(x=matches['match_p_src'][0], y=matches['match_p_src'][1], c='b', s=10)

plt.subplot(2, 2, 4)
plt.imshow(dst)
plt.title('dst with ourliers mp')
plt.scatter(x=matches['match_p_dst'][0], y=matches['match_p_dst'][1], c='b', s=10)

plt.show(block=False)

# Part A - Homography naive
H_naive_perfect_mp = compute_homography_naive(match_perfect['match_p_src'], match_perfect['match_p_dst'])
H_naive_not_perfect_mp = compute_homography_naive(matches['match_p_src'], matches['match_p_dst'])

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(src)
plt.title('original')

plt.subplot(1, 3, 2)
transformed_src_image_perfect_naive = cv2.warpPerspective(src, H_naive_perfect_mp, np.shape(src)[0:2])
plt.imshow(transformed_src_image_perfect_naive)
plt.title('H_naive_perfect_mp')

plt.subplot(1, 3, 3)
transformed_src_image_not_perfect_naive = cv2.warpPerspective(src, H_naive_not_perfect_mp, np.shape(src)[0:2])
# transformed_src_image_not_perfect_naive_affine = cv2.warpAffine(transformed_src_image_not_perfect_naive, M, np.shape(transformed_src_image_not_perfect_naive)[0:2])
plt.imshow(transformed_src_image_not_perfect_naive)
plt.title('H_naive_not_perfect')

plt.show(block=False)

# Part B - homography ransac
## Caculate homography using perfect match points
# fit_percent, dist_mse = test_homography(H_perfect, match_perfect['match_p_src'], match_perfect['match_p_dst'], max_err=2)
H = compute_homography(match_perfect['match_p_src'], match_perfect['match_p_dst'], inliers_percent=0.5, max_err=5)

# Part C - panorama
max_err = 25
inliers_percent = 0.8

img_pan = panorama(src, dst,matches['match_p_src'], matches['match_p_dst'], inliers_percent, max_err)
