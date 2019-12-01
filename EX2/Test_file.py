## Import needed packages

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from ex2_functions import *

## Load data
src = cv2.imread('Input_Files/src.jpg')
dst = cv2.imread('Input_Files/dst.jpg')
matches = sio.loadmat('Input_Files/matches.mat')
match_perfect = sio.loadmat('Input_Files/matches_perfect.mat')


## Display images + perfect matching points
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

## Caculate homography using perfect match points
H_perfect = compute_homography_naive(match_perfect['match_p_src'], match_perfect['match_p_dst'])
fit_percent, dist_mse = test_homography(H_perfect, match_perfect['match_p_src'], match_perfect['match_p_dst'], max_err=2)
img_pan = panorama(src, dst, mp_src, mp_dst, fit_percent, max_err)


print('hi')