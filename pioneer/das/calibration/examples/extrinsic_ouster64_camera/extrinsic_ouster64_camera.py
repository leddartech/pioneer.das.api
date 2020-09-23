'''Created on 30/03/2020
olivier.blondeau-fournier@leddartech.com

A simple (and intuitive) guide using opencv for extrinsics between camera and a lidar: here Ouster64.
The idea is to use corners, both in camera and ouster, and to deduce rotation R and translation T from Lidar ref to Cam ref.

Useful concepts:
Lidar ref: 3d point X0,Y0,Z0
Camera ref: also 3d point X1,Y1,Z1
Camera uv ref/focal plane: u,v coordinate in a image.

Usually, if you know the rotation matrix R, and translation T from Lidar to Camera ref, and
if you know the intrinsic matrix K of the camera, you could use 
s*(u,v,1) = K . [R|T] . (X0,Y0,Z0,1)
However, since we also have distortion coeffs, we use:

cv2.projectPoints(XYZ0, rotation_vec, translation_vec, K, distortion)

In this script, we want to determine the extrinsic matrix R, and translation T, so that a point in Lidar ref
is mapped to Camera ref; the only action remaining is projecting points from Camera ref, to Camera uv:

cv2.projectPoints(XYZ1, np.zeros((3,1)), np.zeros((3,1)), K, distortion)

'''
from pioneer.das.calibration.examples.intrinsic_camera_opencv import intrinsic_camera_opencv as I 

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm

import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
sns.set()

#For this method to work, we will  need a prior transformation (or a initial guess)
class Checkerboard():
	def __init__(self, 
					pattern=(4,3), 
					square_length=0.145, 
					criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)):
		self.pattern = pattern #(nx,ny)
		self.square_length = square_length
		self.criteria = criteria
		self.points = self._compute_points()
		self.n_corners = self.pattern[0]*self.pattern[1]
		self.n_tiles = int((self.pattern[0]+1)*(self.pattern[1]+1)/2)
	
	def _compute_points(self):
		points_ = np.zeros((self.pattern[0] * self.pattern[1], 3), dtype=np.float32)
		points_[:, :2] = np.mgrid[0:self.pattern[0], 0:self.pattern[1]].T.reshape(-1, 2)
		points_ *= self.square_length
		return points_

class ProtoCameraToLidar():
	'''This is a basic class if you know a prior calib from Camera ref to
		Lidar ref using the checkerboard.  For instance, one can select manually one pair of frames,	
		solve svd and get the transformation matrix/translation. 

	'''
	def __init__(self, 
					checkerboard, 
					mtx, 
					dist, 
					R, 
					T, 
					view, 
					epsilons):
		self.checkerboard = checkerboard
		self.mtx = mtx
		self.dist = dist
		self.R = R #rotation matrix from Cam ref to Lidar ref
		self.T = T #translation vector from Cam ref to Lidar ref
		self.view = view
		self.epsilons = epsilons
	
	def map_uv_to_pts3d(self, cam_uv_pts):
		'''Retuns both pts3d_cam and pts3d_lidar.
		'''
		#extract 3d pose:
		_,  r_, t_ = cv2.solvePnP(self.checkerboard.points, cam_uv_pts, self.mtx, self.dist)
		#transform r_ vec to mat:
		r_, _ = cv2.Rodrigues(r_)
		#maps back those checkerboard.points to 3d cam ref:
		pts3d_cam = mapto(self.checkerboard.points, r_, t_)
		#and to lidar:
		pts3d_lidar = mapto(pts3d_cam, self.R, self.T)
		return pts3d_cam, pts3d_lidar
	
def mapto(pts, R,T):
	'''
	pts - [N,3]
	'''
	return (R.dot(pts.T) + T).T

def load_intrinsic_camera(pkl_file):
	'''return K-matrix and distortion
	'''
	with open(pkl_file, 'rb') as g:
		calib_data = pickle.load(g)
	mtx = calib_data['matrix']
	dist = calib_data['distortion']
	return mtx, dist 

def get_camera_corners(img, checkerboard, show):
	'''return uv coordinate of the corners (in camera focal ref)
	'''
	return I.get_corners(img, checkerboard.pattern, checkerboard.criteria, show)

def get_mask_lidar_from_prior_pts(X_prior, X_lidar, epsilon_x, epsilon_y, epsilon_z):
	''' both are [*,3] for some *
	'''
	x_min, x_max = X_prior[:,0].min(), X_prior[:,0].max()
	y_min, y_max = X_prior[:,1].min(), X_prior[:,1].max()
	z_min, z_max = X_prior[:,2].min(), X_prior[:,2].max()

	mask = (X_lidar[:,0]>= x_min - epsilon_x) * (X_lidar[:,0]<= x_max + epsilon_x)
	mask *= (X_lidar[:,1]>= y_min - epsilon_y) * (X_lidar[:,1]<= y_max + epsilon_y)
	mask *= (X_lidar[:,2]>= z_min - epsilon_z) * (X_lidar[:,2]<= z_max + epsilon_z)
	return mask

def get_lidar_corners(pts, A, priorCamLidar, corners_cam_uv, 
							show=False, tolerance_eps=0.025, 
							tolerance_nedges=8):
	''' pts - lidar 3d points
		A - amplitude

	'''
	checkerboard = priorCamLidar.checkerboard
	r = checkerboard.square_length
	epsilons = priorCamLidar.epsilons
	pts3d_cam_, pts3d_lidar_prior = priorCamLidar.map_uv_to_pts3d(corners_cam_uv)
	mask = get_mask_lidar_from_prior_pts(pts3d_lidar_prior, pts, 
											epsilon_x=(epsilons[0]*r), 
											epsilon_y=(epsilons[1]*r), 
											epsilon_z=(epsilons[2]*r))
	Xp, Ap = pts[mask], A[mask]
	corners_ = find_lidar_chess_corners(Xp, Ap, threshold_amplitude=np.median(Ap), checkerboard=checkerboard)
	corners_ = sort_corner_pts(corners_, checkerboard, priorCamLidar.view)
	ret = coherence_criteria_3dcorners(corners_, checkerboard, tolerance_eps, tolerance_nedges)
	if show:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], s=10, c=Ap, marker='d', label='Points from Lidar')
		ax.scatter(corners_[:,0], corners_[:,1], corners_[:,2], s=50,color='green' , marker='o', label='Detected corners')
		ax.scatter(corners_[0,0], corners_[0,1], corners_[0,2], s=50,color='yellow' , marker='o')
		ax.plot(corners_[:,0], corners_[:,1], corners_[:,2], color='r')
		ax.set_xlabel('RET value: '+ str(ret))
		plt.legend()
		plt.show()
	
	return ret, corners_, pts3d_cam_

def distance_point_plane(X,a,b,c,d):
	"""Distance to plane: Plane has equation::
		P: a*x+b*y+c*z+d = 0
	"""
	return np.abs(a*X[:,0]+b*X[:,1]+c*X[:,2]+d)/(a**2+b**2+c**2)**0.5

def iterate_over_proj_plane(X, max_iter=5, score=0.96, epsilon=0.15):
	"""find the best plane until one criteria is obtained
		
		return: mask, (xx,yy,zz) of grid plane
		Note that by default, we would like to assume that the plane is almost perpendicular to x-y plane
	"""
	reg = LinearRegression()
	Y = X.copy()
	mask = np.ones(Y.shape[0], bool)
	for i in range(max_iter):
		x_fit, y_fit = Y[mask][:,1:], Y[mask][:,0:1]
		reg.fit(x_fit,y_fit)
		score_ = reg.score(x_fit, y_fit)
		if score_ > score:
			break
		d_to_plane = distance_point_plane(Y, -1, reg.coef_[0][0], reg.coef_[0][1], reg.intercept_)
		d_med = np.median(d_to_plane)
		if d_med < epsilon:
			d_med = epsilon
		
		mask_ = (d_to_plane < np.median(d_med)).astype(bool)
		mask *= mask_
	
	yy, zz = np.meshgrid(np.linspace(Y[mask][:,1].min(),Y[mask][:,1].max(), 20) , 
					 np.linspace(Y[mask][:,2].min(),Y[mask][:,2].max(), 20))
	xx = yy * reg.coef_[0][0] + zz * reg.coef_[0][1] + reg.intercept_
	P = (-1, reg.coef_[0][0], reg.coef_[0][1], reg.intercept_[0])
	return mask, P, (xx,yy,zz), score_


def project_points_to_plane(X,P):
	""" X = [N,3]
		P = (a,b,c,d) as in a*x+by+c*z+d=0
	"""
	t = -(P[0]*X[:,0]+P[1]*X[:,1]+P[2]*X[:,2]+P[3])/(P[0]**2+P[1]**2+P[2]**2)
	Y = X.copy()
	Y[:,0]=X[:,0]+t*P[0]
	Y[:,1]=X[:,1]+t*P[1]
	Y[:,2]=X[:,2]+t*P[2]
	return Y

def find_square_center(X,A,threshold, checkerboard):
	""" clusterize the square pattern Black/White, given by the amplitude threshold.
	"""
	clustering_w = KMeans(n_clusters=checkerboard.n_tiles, random_state=0)
	clustering_b = KMeans(n_clusters=checkerboard.n_tiles, random_state=0)
	mask0 = (A>threshold).astype(bool)
	Yw = X[mask0]
	Yb = X[~mask0]
	clustering_w.fit(Yw)
	clustering_b.fit(Yb)
	return clustering_w.cluster_centers_, clustering_b.cluster_centers_

def get_dual_grid_points(X,radius):
	"""obtain all the points of the dual grid, this is done by iteratively considering nearest neib.
	"""
	dualX = []
	tree = BallTree(X, leaf_size=5)
	for i in range(len(X)):
		ids = tree.query_radius(X[i:i+1], r=radius)
		for j in ids[0]:
			if j!=i:
				t = X[j:j+1]-X[i:i+1]
				dualX.append(X[i:i+1]+0.5*t)
	
	return np.vstack(dualX)

def find_lidar_chess_corners(X, A, threshold_amplitude, checkerboard):
	mask, P, (xx,yy,zz), score = iterate_over_proj_plane(X, max_iter=10, score=0.95, epsilon=0.15)
	Y,B = X[mask], A[mask]
	Y = project_points_to_plane(Y,P)
	Yw, Yb = find_square_center(Y,B, threshold_amplitude, checkerboard)
	radius = 2**0.5 * 1.25 * checkerboard.square_length
	dualYb =  get_dual_grid_points(Yb, radius)
	dualYw =  get_dual_grid_points(Yw, radius)
	dualY = np.vstack((dualYb,dualYw))
	clustering = KMeans(n_clusters=checkerboard.n_corners, random_state=0)
	clustering.fit(dualY)
	return clustering.cluster_centers_

def sort_corner_pts(pts, checkerboard, view='c'):
	'''suppose left to right, from top to bottom
	'''
	n_cols = checkerboard.pattern[0] #4
	n_rows = checkerboard.pattern[1] #3
	ids0 = np.argsort(pts[:,2])[::-1] #sort top-bottom
	pts = pts[ids0]
	for i in range(n_rows):
		if view is 'c':
			ids_ = np.argsort(pts[i*n_cols:(i+1)*n_cols][:,1])[::-1]
		elif view is 'l':
			ids_ = np.argsort(pts[i*n_cols:(i+1)*n_cols][:,0])
		elif view is 'r':
			ids_ = np.argsort(pts[i*n_cols:(i+1)*n_cols][:,0])[::-1]

		pts[i*n_cols:(i+1)*n_cols] = pts[i*n_cols:(i+1)*n_cols][ids_]
	return pts[::-1]#from bottom to top

def coherence_criteria_3dcorners(X, checkerboard, tolerance_eps, tolerance_nedges):
	'''Try to test if the points X are valid corners of a checkerboard
	'''
	Y = X.reshape((checkerboard.pattern[1], checkerboard.pattern[0], 3))
	h_error = []
	v_error = []

	for i in range(checkerboard.pattern[0]-1):
		h_error.append(lj_error(Y[:,i,:], Y[:,i+1,:]))
	for j in range(checkerboard.pattern[1]-1):
		v_error.append(lj_error(Y[j,:,:], Y[j+1,:,:]))

	h_error = np.abs(np.hstack(h_error) - checkerboard.square_length)
	v_error = np.abs(np.hstack(v_error) - checkerboard.square_length)

	bad_edges = np.sum(h_error>tolerance_eps) + np.sum(v_error>tolerance_eps)
	return (bad_edges <= tolerance_nedges)

def euclidean_transform_3D(A, B):
	'''
		A,B - Nx3 matrix
		return:
			R - 3x3 rotation matrix
			t = 3x1 column vector
	'''
	assert len(A) == len(B)

	N = A.shape[0]; 
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))
	H = np.dot(np.transpose(AA), BB)
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T , U.T)
   
	# handle svd sign problem
	if np.linalg.det(R) < 0:
		Vt[2,:] *= -1
		R = np.dot(Vt.T, U.T)

	t = -np.dot(R,centroid_A.T) + centroid_B.T
	return R, t

def lj_error(a, b, j=2.0):
	'''a,b are 3d pts (N,3)
	'''
	return ((a[:,0]-b[:,0])**j+(a[:,1]-b[:,1])**j+(a[:,2]-b[:,2])**j)**(1/j)


# if __name__ == '__main__':
	#Exemple of use: see the jupyter notebook

	






