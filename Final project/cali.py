
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# LOAD IMAGES
imgs_left = glob.glob(r'.\Stereo_calibration_images\left*.png')
imgs_right = glob.glob(r'.\Stereo_calibration_images\right*.png')

assert imgs_right, imgs_left
assert (len(imgs_right) == len(imgs_left))
n_images = len(imgs_right)
imgs_right.sort()
imgs_left.sort()

# %% INITIALIZE CHECKERBOARD OBJECT POINTS
nb_vertical = 9
nb_horizontal = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # vector of vectors of calibration pattern points in the calibration pattern coordinate space
imgpoints_left = [] # 2d points in image plane.
imgpoints_right = [] # 2d points in image plane.

# %% EXTRACT CHECKERBOARD CORNERS (AND DISPLAY THEM)
for i in range(n_images):
    img_left = cv2.imread(imgs_left[i])
    img_right = cv2.imread(imgs_right[i])
    gray_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, patternSize = (nb_vertical, nb_horizontal))
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, patternSize = (nb_vertical, nb_horizontal))
    # If found, add object points, image points (after refining them)
    if ret_left == True and ret_right == True:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        # Draw and display the corners
        img_left = cv2.drawChessboardCorners(img_left, (nb_vertical,nb_horizontal), corners_left,ret_left)
        img_right = cv2.drawChessboardCorners(img_right, (nb_vertical,nb_horizontal), corners_right,ret_right)
        cv2.putText(img_left, "Left Camera", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img_right, "Right Camera", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        img_l_r = np.vstack((img_left,img_right))
        windowname_1 = "Calibration Pattern (Left and Right Camera)"
        cv2.namedWindow(windowname_1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname_1, 600, 600)
        cv2.imshow(windowname_1,img_l_r)
        cv2.waitKey(5)
cv2.destroyAllWindows()

# %% STEREOCALIBRATE - COMPUTE (INTRINSIC) CAMERA MATRICES, DISTORTION COEFFICIENTS, ROTATION AND TRANSLATION BETWEEN THE TWO CAMERAS AND REPROJECTION ERROR
assert (img_left.shape[:2] == img_right.shape[:2])
h, w = img_left.shape[:2]

term_crit_sc = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags_sc = cv2.CALIB_RATIONAL_MODEL
ret_stereo,  mtx_left, dist_left, mtx_right, dist_right, mtx_R, mtx_T, mtx_E, mtx_F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, None, None, None, None, (w,h), flags=flags_sc, criteria=term_crit_sc)
print(mtx_left, dist_left)

# %% STEREORECTIFY - COMPUTE AND SAVE THE RECTIFICATION TRANSFORM AND PROJECTION MATRIX OF THE 2 CAMERAS, USING THE MATRICES COMPUTED BY STEREOCALIBRATE

mtx_R_left, mtx_R_right, mtx_P_left, mtx_P_right, mtx_Q, roi_rec_left, roi_rec_right = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, (w,h), mtx_R, mtx_T, alpha=0)

# %% COMPUTE UNDISTORTION AND RECTIFICATION TRANSFORMATION MAP
map1x, map1y = cv2.initUndistortRectifyMap(mtx_left,dist_left,mtx_R_left,mtx_P_left,(w,h),cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_right,dist_right,mtx_R_right,mtx_P_right,(w,h),cv2.CV_32FC1)