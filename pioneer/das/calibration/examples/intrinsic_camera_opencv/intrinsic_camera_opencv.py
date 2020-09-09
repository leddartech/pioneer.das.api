from pioneer.das.api.platform import Platform

from tqdm import tqdm

import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set()

'''Created on 30/03/2020
olivier.blondeau-fournier@leddartech.com

A simple and intuitive guide using opencv for camera intrinsic calibration.
'''

def _pdist(p1, p2):
	return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def _get_half_corner_distance(checkerboard, corners):
	min_distance = float("inf")
	n_cols = checkerboard[0]
	n_rows = checkerboard[1]
	for row in range(n_rows):
		for col in range(n_cols - 1):
			index = row*n_rows + col
			min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
	for row in range(n_rows - 1):
		for col in range(n_cols):
			index = row*n_rows + col
			min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + n_cols, 0]))
	radius = int(math.ceil(min_distance * 0.5))
	return radius

def get_corners( img, 
				 checkerboard, 
				 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1), 
				 show_img_result=False):
	'''Finds the pixell positions of the corners, using a sub-pixell refinment.
	'''
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, 
											 checkerboard, 
											 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
	
	if ret == True:
		radius = _get_half_corner_distance(checkerboard, corners)
		corners2 = cv2.cornerSubPix(gray, 
									corners, 
									(radius,radius),
									(-1,-1), 
									criteria)
		if show_img_result:
			gray = cv2.drawChessboardCorners(gray, checkerboard, corners2, ret)
			cv2.imshow('gray',gray)
			cv2.waitKey(0)

		return ret, corners2
	else:
		return ret, corners

def get_calibration(objpoints, imgpoints, img_shape):
	return cv2.calibrateCamera(objpoints, 
								imgpoints,
								img_shape,
								None,
								None)

def iterate_over_platform( dataset, 
						   camera_key, 
						   checkerboard, 
						   criteria, 
						   frames=None,
						   show_img_result=False):
	'''Do a callibration over a complete dataset, or provided a list of frames.
	'''
	pf = Platform(dataset)
	objpoints = []
	imgpoints = []
	objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
	objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

	frames = frames if frames is not None else np.arange(len(pf[camera_key])) 
	for mu in tqdm(frames):
		sample_img = np.copy(pf[camera_key][mu].raw)
		ret, corners = get_corners(sample_img, checkerboard, criteria, show_img_result)
		
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

	if show_img_result:
		cv2.destroyAllWindows()
	
	sh = pf[camera_key][0].raw.shape
	img_shape = (sh[1], sh[0])

	return get_calibration(objpoints, imgpoints, img_shape)

def display_undistorded_corners(img, mtx, dist, checkerboard):
    ret, corners = get_corners(img, checkerboard)
    img = cv2.drawChessboardCorners(img.copy(), checkerboard, corners, ret)
    cv2.imshow('Img',img)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    imgund=cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow('Undidstorded', imgund)

def get_undistorded_img(img, mtx, dist, checkerboard, alpha=1):
    ret, corners = get_corners(img, checkerboard)
    img = cv2.drawChessboardCorners(img.copy(), checkerboard, corners, ret)
    h,  w = img.shape[:2]
    if alpha is not None:
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),alpha,(w,h))
        imgund=cv2.undistort(img, mtx, dist, None, newcameramtx)
        return imgund, (roi, newcameramtx)
    else:
        imgund=cv2.undistort(img, mtx, dist, None, None)
        return imgund


if __name__ == '__main__':
	#Exemple of use (see also the jupyter notebook):
	dataset = '/nas/pixset/calibration/intrinsic_camera/v3_20200326/'
	camera_key = 'flir_bfc_intr-calib-img'
	checkerboard = (13,10)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
	show_img_result = False

	ret, mtx, dist, rvecs, tvecs = iterate_over_platform(dataset, 
															camera_key, 
															checkerboard, 
															criteria, 
															show_img_result)
	
	print("Results for camera calibration : \n")
	print("Camera matrix : \n")
	print(mtx)
	print("Distorsion coeffs: \n")
	print(dist)
	#print("rvecs : \n")
	#print(rvecs)
	#print("tvecs : \n")
	#print(tvecs)






