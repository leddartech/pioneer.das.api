# tentative de calibration LCA3 - Cam Left - Cam Right

# -*- coding: utf-8 -*-

# s'utilise après le script camLeddarHomologs.py
# prend trois nuages de points et essaye de trouver la transformation rigide entre eux en utilisqnt egqlement la boucle des 3 transformations


from mpl_toolkits import mplot3d
from numpy import dtype, argsort
from scipy.optimize import minimize
from transforms3d import euler, axangles

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import transforms3d

class PointCloudTransformationSolver:

    def __init__(self):
        self.m = 3
        self.n = 4

    def objective(self, x, Pleddar, Pcam):
        nbPts = len(Pleddar[1, :])
        PcamHat = np.zeros((self.m, nbPts))
        tmp = x.reshape(3, 4)
        for i in range(0, nbPts):
            PcamHat[:, i] = tmp.dot(Pleddar[:, i])

        Err = np.zeros((self.m, nbPts))
        square = np.zeros(nbPts)
        for i in range(0, len(Pleddar[1, :])):
            Err[:, i] = PcamHat[:, i] - Pcam[0:3, i]
            square[i] = Err[:, i].dot(Err[:, i])
        res = np.sum(square)
        return res

    def linearSolve(self, Pleddar, Pcam):
        
        ptClR = Pleddar.T[:, 0:3]
        ptCrR = Pcam.T[:, 0:3]

        ptCrRc = ptCrR-np.mean(ptCrR, axis=0)[None, :]
        ptClRc = ptClR-np.mean(ptClR, axis=0)[None, :]

        cov = ptCrRc.T.dot(ptClRc) / ptCrRc.shape[1]
        # print(cov)
        u, s, v = np.linalg.svd(cov)

        d = (np.linalg.det(u) * np.linalg.det(v)) < 0.0

        if d:
            s[-1] = -s[-1]
            u[:, -1] = -u[:, -1]
        r = u.dot(v)
        # print(r)
        t = np.mean(ptCrR.T-r.dot(ptClR.T), axis=1)

        Tr = np.hstack((r, t[:, None]))

        return Tr


class RansacCalibration:
    """ 3D points registration using RANSAC algorithm to deal with outlier. """

    def __init__(self, nb_random_sampling, distance_threshold):
        """ constructor

            :param nb_random_sampling: number of random sampling to execute
            :param distance_threshold: point distance error threshold to concider the point valide for the model
        """
        self.nb_random_sampling = nb_random_sampling
        self.distance_threshold = distance_threshold
        self.solver = PointCloudTransformationSolver()

    def count_point_ok_with_model(self, Tr):
        """ check the number of point repect the distance_threshold condition

            :param Tr: transformation to test
            :return : np array list of index respect the threshold
        """
        leddar_points_in_camera = Tr.dot(self.leddar_points)
        erreur = self.camera_points - leddar_points_in_camera
        ErreurDistance = np.linalg.norm(erreur, axis=0)
        return np.where(ErreurDistance < self.distance_threshold)[0]

    def calibrate(self, leddar_points, camera_points):
        """ calibration with ransac

            :param leddar_points: leddar points numpy array 4*N (homogeneous notation)
            :param camera_points: Camera points numpy array 4*N (homogeneous notation)
            :return : Tr = homogeneous transformation matrix numpy array 4x4
        """
        self.leddar_points = leddar_points
        self.camera_points = camera_points
        self.nb_points = self.leddar_points.shape[1]

        self.sampling_nb_point_result = np.zeros(
            self.nb_random_sampling, dtype=np.float64)
        # numpy array list contain the point index selected ramdomly for each sample
        self.sampling_index_selected = []

        self.solver = PointCloudTransformationSolver()

        for i in range(self.nb_random_sampling):
            # random sample index selection, we need minimum 4 points for the Least squares method
            index_selection = np.random.choice(range(self.nb_points), 4)

            # compute linear calibration
            x = self.solver.linearSolve(
                leddar_points[:, index_selection], camera_points[:, index_selection])
            Tr = np.eye(4)
            Tr[0:3, 0:4] = x.reshape(3, 4)
            index_points_ok_with_threshold = self.count_point_ok_with_model(Tr)

            # store them
            self.sampling_index_selected.append(index_points_ok_with_threshold)
            self.sampling_nb_point_result[i] = index_points_ok_with_threshold.shape[0]

        idx_sorted = np.argsort(self.sampling_nb_point_result)

        # we keep the best inlier points
        index_inlier_best_points = self.sampling_index_selected[idx_sorted[-1]]
        # compute the trqnsformation
        x = self.solver.linearSolve(
            leddar_points[:, index_inlier_best_points], camera_points[:, index_inlier_best_points])
        Tr = np.eye(4)
        Tr[0:3, 0:4] = x.reshape(3, 4)
        # and the reprojection error
        error = self.solver.objective(
            x, leddar_points[:, index_inlier_best_points], camera_points[:, index_inlier_best_points])

        return Tr, index_inlier_best_points, error / index_inlier_best_points.shape[0]



def axangle4to3(ax):
    normal = ax[0]
    angle = ax[1]
    return normal*angle


def axangle3to4(ax):
    angle = np.linalg.norm(ax)
    if(angle != 0):
        return (ax/angle, angle)
    else:
        return ([0.0, 0.0, 1.0], 0)


def axangleX0init(mat):

    t = mat[0:3, 3]
    r = mat[0:3, 0:3]

    ax = transforms3d.axangles.mat2axangle(r)
    ax3 = axangle4to3(ax)
    x0 = np.zeros((6), dtype=np.float64)
    x0[0:3] = t
    x0[3:6] = ax3

    # print('axangleX0init', mat,ax,ax3, x0)
    return x0


def x2mat(x):
    x0 = np.zeros((3, 4), dtype=np.float64)
    t = x[0:3]
    x0[0:3, 3] = t
    ax = x[3:6]
    axx, angle = axangle3to4(ax)
    # print('axx, angle', axx, angle)
    x0[0:3, 0:3] = transforms3d.axangles.axangle2mat(axx, angle)
    return x0.ravel()


def state2Affine(x):
    tmp = np.eye(4, dtype=np.float64)
    t = x[0:3]
    ax = x[3:6]
    ax, angle = axangle3to4(ax)
    # print('axx, angle2', ax, angle)
    R = transforms3d.axangles.axangle2mat(ax, angle)
    tmp[0:3, 0:3] = R
    tmp[0:3, 3] = t
    return tmp




def inverseTransform(t):
    tr = np.eye(4)
    rr = t[0:3, 0:3].transpose()
    tt = t[0:3, 3]
    tr[0:3, 0:3] = rr
    tr[0:3, 3] = -rr.dot(tt)
    return tr


def error_projection(Tin2out, ptsIn, ptsOut, meanPoint=False):
    ptsIninOut = Tin2out.dot(ptsIn)
    error = ptsIninOut-ptsOut
    squareError = np.sum(error * error, axis=0)
    # return np.mean(squareError)
    if meanPoint:
        return np.mean(squareError)
    else:
        return np.sum(squareError)


def objective_boucle(x):
    #print(x)
    x_Cl2LCA = x[0:6]
    x_Cr2LCA = x[6:12]
    x_Cl2Cr = x[12:18]

    T_Cl2LCA = state2Affine(x_Cl2LCA)
    T_Cr2LCA = state2Affine(x_Cr2LCA)
    T_Cl2Cr = state2Affine(x_Cl2Cr)

    errorCl2LCA = error_projection(
        T_Cl2LCA, ptsCameraLeft_ransac, ptsLeddarLeft_ransac)
    errorCr2LCA = error_projection(
        T_Cr2LCA, ptsCameraRight_ransac, ptsLeddarRight_ransac)
    errorCl2Cr = error_projection(
        T_Cl2Cr, ptscamLeft_ransac, ptscamRight_ransac)

    T_loop = inverseTransform(T_Cl2LCA).dot((T_Cr2LCA).dot(T_Cl2Cr))
    #print('T_loop\n{}'.format(T_loop))
    errorBoucle = error_projection(T_loop, ptscamLeft_ransac, ptscamLeft_ransac)
    #print('{}->{}'.format([errorCl2LCA, errorCr2LCA, errorCl2Cr, errorBoucle], errorCl2LCA+ errorCr2LCA+ errorCl2Cr+ errorBoucle))
    return errorCl2LCA + errorCr2LCA + errorCl2Cr + errorBoucle


def getEulerAngleZYX(R):
    angle = np.zeros(3, dtype=np.float64)

    angle[1] = math.asin(R[2, 0])  # math.asin(R[0, 2])
    # math.acos(R[0, 0] / math.cos(angle[1]))
    angle[2] = math.atan2(-R[1, 0], R[0, 0])
    # math.acos(R[2, 2] / math.cos(angle[1]))
    angle[0] = math.atan2(-R[2, 1], R[2, 2])
    return -angle


if __name__ == '__main__':
    #np.random.seed(0)
    parser = argparse.ArgumentParser(
        description='Process the extrinsic calibration between the 2 cameras flir and LCA3, loop constraint.')
    parser.add_argument('homologsPoints3dFolder',
                        help='3d points folder, Nx7 file for cam left, cam right and lca.')
    parser.add_argument('-i', '--ransacIterations', type=int,
                        default=5000, help='Number ransac iterations')
    parser.add_argument('-t', '--ransacThreshold', type=float,
                        default=0.15, help='Ransac threshold distance')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output calibration file')
    parser.add_argument('-q', '--quiet', default=False, action='store_true',
                        help='display console results')
    parser.add_argument('-p', '--plot', default=False, action='store_true',
                        help='Plot point')
    # parser.add_argument('-d', '--dump', action='store_true', help='dump 3d output points xyz files in camera frame, usefull for cloudcompare')

    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as err:
        print(err)
        exit()

    verbose = not args.quiet

    if verbose:
        print('Load 3d points files...')

    # left camera to lca
    pts = np.loadtxt(os.path.join(
        args.homologsPoints3dFolder, 'h_left_lca3.csv'))
    idxLeft = pts[:, 0]
    ptsLeddarLeft = pts[:, 4:7]
    ptsCameraLeft = pts[:, 1:4]

    # right camera to lca
    pts = np.loadtxt(os.path.join(
        args.homologsPoints3dFolder, 'h_right_lca3.csv'))
    idxRight = pts[:, 0]
    ptsLeddarRight = pts[:, 4:7]
    ptsCameraRight = pts[:, 1:4]

    # left to right cameras
    pts = np.loadtxt(os.path.join(
        args.homologsPoints3dFolder, 'h_left_right.csv'))
    ptscamLeft = pts[:, 1:4]
    ptscamRight = pts[:, 4:7]

    # convert to homogeneous coordinates points
    ptsLeddarLeft = np.concatenate([ptsLeddarLeft, np.ones(
        (ptsLeddarLeft.shape[0], 1), dtype=ptsLeddarLeft.dtype)], axis=1).T
    ptsCameraLeft = np.concatenate([ptsCameraLeft, np.ones(
        (ptsCameraLeft.shape[0], 1), dtype=ptsCameraLeft.dtype)], axis=1).T

    ptsLeddarRight = np.concatenate([ptsLeddarRight, np.ones(
        (ptsLeddarRight.shape[0], 1), dtype=ptsLeddarRight.dtype)], axis=1).T
    ptsCameraRight = np.concatenate([ptsCameraRight, np.ones(
        (ptsCameraRight.shape[0], 1), dtype=ptsCameraRight.dtype)], axis=1).T

    ptscamRight = np.concatenate([ptscamRight, np.ones(
        (ptscamRight.shape[0], 1), dtype=ptscamRight.dtype)], axis=1).T
    ptscamLeft = np.concatenate([ptscamLeft, np.ones(
        (ptscamLeft.shape[0], 1), dtype=ptscamLeft.dtype)], axis=1).T

    # init the minimize process with the ransac transformation
    if verbose:
        print('Ransac computing...')
    ransacInit = RansacCalibration(args.ransacIterations, args.ransacThreshold)
    TrRansacCl2LCA, indexRansacCl2LCA, errorRansacCl2LCA = ransacInit.calibrate(
        ptsCameraLeft, ptsLeddarLeft)
    if verbose:
        print('left to lca')
    TrRansacCr2LCA, indexRansacCr2LCA, errorRansacCr2LCA = ransacInit.calibrate(
        ptsCameraRight, ptsLeddarRight)
    if verbose:
        print('left to righ')
    TrRansacCl2Cr, indexRansacCl2Cr, errorRansacCl2Cr = ransacInit.calibrate(
        ptscamLeft, ptscamRight)
    if verbose:
        print('left to right')
    # keep only the ransac selected points
    ptsLeddarLeft_ransac = ptsLeddarLeft[:, indexRansacCl2LCA]
    ptsCameraLeft_ransac = ptsCameraLeft[:, indexRansacCl2LCA]

    ptsLeddarRight_ransac = ptsLeddarRight[:, indexRansacCr2LCA]
    ptsCameraRight_ransac = ptsCameraRight[:, indexRansacCr2LCA]

    ptscamRight_ransac = ptscamRight[:, indexRansacCl2Cr]
    ptscamLeft_ransac = ptscamLeft[:, indexRansacCl2Cr]

    # convert each ransac transformation to state vector [Tx,Ty,Tz,Ax,Ay,Az] : translation and angle angle rotation
    xTrRansacCl2LCA = axangleX0init(TrRansacCl2LCA)
    xTrRansacCr2LCA = axangleX0init(TrRansacCr2LCA)
    xTrRansacCl2Cr = axangleX0init(TrRansacCl2Cr)

    # display the ransac transfortation
    if verbose:
        print('Ransac transformations :')
        print('TrRansacCl2LCA:\n{}'.format(TrRansacCl2LCA))
        print('TrRansacCr2LCA:\n{}'.format(TrRansacCr2LCA))
        print('TrRansacCl2Cr:\n{}'.format(TrRansacCl2Cr))
        print('TrRansac_loop:\n{}'.format(inverseTransform(TrRansacCl2LCA).dot(
            (TrRansacCr2LCA).dot(TrRansacCl2Cr))))

    # concatenate the full state vector to optimize
    # [cam left --> LCA3, cam right --> LCA3, cam left --> cam right]
    x0 = np.concatenate((xTrRansacCl2LCA, xTrRansacCr2LCA, xTrRansacCl2Cr))

    if verbose:
        print('')
        print('x0={}'.format(x0))
        print('')
        print('Starting minimization...')
        print('')

    # , 'ftol':1e-8,'eps':1.5e-6})#, bounds=bnds, constraints=cons)
    solution = minimize(objective_boucle, x0,
                        method='SLSQP', options={'disp': True})
    print(solution)

    xResult = solution.x

    if verbose:
        print('')
        print('found x:\n{}'.format(xResult))

    # split state vector for each transformation
    x_Cl2LCA = xResult[0:6]
    x_Cr2LCA = xResult[6:12]
    x_Cl2Cr = xResult[12:18]

    # convert each state vector to homogeneous matrix transformation
    T_Cl2LCA = state2Affine(x_Cl2LCA)
    T_Cr2LCA = state2Affine(x_Cr2LCA)
    T_Cl2Cr = state2Affine(x_Cl2Cr)
    T_loop = inverseTransform(T_Cl2LCA).dot((T_Cr2LCA).dot(T_Cl2Cr))

    if verbose:
        print('T_Cl2LCA:\n{}'.format(T_Cl2LCA))
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(T_Cl2LCA))))
        print('axis angles:{}'.format(
            axangles.mat2axangle(T_Cl2LCA[0:3, 0:3])))
        print('T_Cr2LCA:\n{}'.format(T_Cr2LCA))
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(T_Cr2LCA))))
        print('axis angles:{}'.format(
            axangles.mat2axangle(T_Cr2LCA[0:3, 0:3])))
        print('T_Cl2Cr:\n{}'.format(T_Cl2Cr))
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(T_Cl2Cr))))
        print('axis angles:{}'.format(axangles.mat2axangle(T_Cl2Cr[0:3, 0:3])))

        print('T_loop:\n{}'.format(T_loop))
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(T_loop))))
        print('axis angles:{}'.format(
            axangles.mat2axangle(T_loop[0:3, 0:3])))
        print('Norm loop translation:{}'.format(
            np.linalg.norm(T_loop[0:3, 3])))

        # reprojection 
        print('')
        errorCl2LCA = np.sqrt(error_projection(
            T_Cl2LCA, ptsCameraLeft_ransac, ptsLeddarLeft_ransac, True))
        errorCr2LCA = np.sqrt(error_projection(
            T_Cr2LCA, ptsCameraRight_ransac, ptsLeddarRight_ransac, True))
        errorCl2Cr = np.sqrt(error_projection(
            T_Cl2Cr, ptscamLeft_ransac, ptscamRight_ransac, True))
        errorBoucle = np.sqrt(error_projection(T_loop, ptscamLeft_ransac, ptscamLeft_ransac, True))
        print('3d points reprojection mean error [cam left --> LCA3, cam right --> LCA3, cam left --> cam right]:\n[{},{},{}]'.format(
            errorCl2LCA, errorCr2LCA, errorCl2Cr))
        print('left cam 3d points loop reprojection mean error:{}'.format(errorBoucle))

    if args.output is None:
        output_file = os.path.join(
            args.homologsPoints3dFolder, 'extrinsic_loop_left_right_LCA.pkl')
    else:
        output_file = args.output

    with open(output_file, 'wb') as f:
        pickle.dump({'T_Cl2LCA': T_Cl2LCA, 'T_Cl2LCAInv': inverseTransform(T_Cl2LCA), 'T_Cr2LCA': T_Cr2LCA, 'T_Cr2LCAInv': inverseTransform(
            T_Cr2LCA), 'T_Cl2Cr': T_Cl2Cr, 'T_Cl2CrInv': inverseTransform(T_Cl2Cr), 'T_loop': T_loop, 'T_loopInv': inverseTransform(T_loop)}, f)


    if args.plot:
        plt.figure()
        ax = plt.axes(projection='3d')

        plt.title('data in leddar frame')
        
        
        
        ax.plot3D(ptsLeddarLeft[0,:],ptsLeddarLeft[1,:],ptsLeddarLeft[2,:],'.r',label='leddar pts')
        ax.plot3D(ptsLeddarRight[0,:],ptsLeddarRight[1,:],ptsLeddarRight[2,:],'.r')
        point_not_keeped_by_ransac = np.setdiff1d(range(ptsLeddarLeft.shape[1]), indexRansacCl2LCA)
        ax.plot3D(ptsLeddarLeft[0,point_not_keeped_by_ransac],ptsLeddarLeft[1,point_not_keeped_by_ransac],ptsLeddarLeft[2,point_not_keeped_by_ransac],'xr',label='leddar pts ransac rejected')
        point_not_keeped_by_ransac = np.setdiff1d(range(ptsLeddarRight.shape[1]), indexRansacCr2LCA)
        ax.plot3D(ptsLeddarRight[0,point_not_keeped_by_ransac],ptsLeddarRight[1,point_not_keeped_by_ransac],ptsLeddarRight[2,point_not_keeped_by_ransac],'xr')
        
        ptscamLeftInLCA3 = T_Cl2LCA.dot(ptscamLeft)
        ptscamRightInLCA3 = T_Cr2LCA.dot(ptscamRight)
        ax.plot3D(ptscamLeftInLCA3[0,:],ptscamLeftInLCA3[1,:],ptscamLeftInLCA3[2,:],'.b',label='left cam pts')
        point_not_keeped_by_ransac = np.setdiff1d(range(ptscamLeftInLCA3.shape[1]), indexRansacCl2Cr)
        ax.plot3D(ptscamLeftInLCA3[0,point_not_keeped_by_ransac],ptscamLeftInLCA3[1,point_not_keeped_by_ransac],ptscamLeftInLCA3[2,point_not_keeped_by_ransac],'xb',label='left cam pts ransac rejected')

        ax.plot3D(ptscamRightInLCA3[0,:],ptscamRightInLCA3[1,:],ptscamRightInLCA3[2,:],'.g',label='right cam pts')
        point_not_keeped_by_ransac = np.setdiff1d(range(ptscamRightInLCA3.shape[1]), indexRansacCl2Cr)
        ax.plot3D(ptscamRightInLCA3[0,point_not_keeped_by_ransac],ptscamRightInLCA3[1,point_not_keeped_by_ransac],ptscamRightInLCA3[2,point_not_keeped_by_ransac],'xg',label='right cam pts ransac rejected')
        
        ax.legend()
 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    # print(np.linalg.qr(TrRansacCr2Cl[0:3,0:3]))
    # print('Ransac number {} {} {}'.format(indexRansacCr2LCA.shape,indexRansacCl2LCA.shape,indexRansacCr2Cl.shape))
    # print('Ransac number  {}'.format(indexRansacCr2Cl.shape))
    # ptscamLeftInRight = TrRansacCr2Cl.dot(ptscamLeft)
    # print(ptscamLeft.T[indexRansacCr2Cl,0:3].shape)
    # tr=ransacInit.solver.linearSolve(ptscamLeft[:,indexRansacCr2Cl],ptscamRight[:,indexRansacCr2Cl]).reshape(3 ,4)
    # trr = np.eye(4)
    # trr[0:3,0:4] = tr
    # print(trr)
    # print(ransacInit.count_point_ok_with_model(trr))
    # print(len(ransacInit.count_point_ok_with_model(trr)))

    # plt.figure()
    # ax = plt.axes(projection='3d')

    # ax.plot3D(ptscamRight.T[indexRansacCr2Cl,0], ptscamRight.T[indexRansacCr2Cl,1], ptscamRight.T[indexRansacCr2Cl,2], '.r', label='camr')
    # ax.plot3D(ptscamLeft.T[indexRansacCr2Cl,0], ptscamLeft.T[indexRansacCr2Cl,1], ptscamLeft.T[indexRansacCr2Cl,2], '.g', label='caml')
    # for j in indexRansacCr2Cl:

    #     ax.plot3D(np.array([ptscamRight.T[j,0],ptscamLeft.T[j,0]]),np.array([ptscamRight.T[j,1],ptscamLeft.T[j,1]]),np.array([ptscamRight.T[j,2],ptscamLeft.T[j,2]]),'b')

    # plt.legend()

    # plt.figure()
    # ax = plt.axes(projection='3d')

    # ax.plot3D(ptscamRight.T[indexRansacCr2Cl,0], ptscamRight.T[indexRansacCr2Cl,1], ptscamRight.T[indexRansacCr2Cl,2], '.r', label='camr')
    # ax.plot3D(ptscamLeftInRight.T[indexRansacCr2Cl,0], ptscamLeftInRight.T[indexRansacCr2Cl,1], ptscamLeftInRight.T[indexRansacCr2Cl,2], '.g', label='caml')
    # for j in indexRansacCr2Cl:

    #     ax.plot3D(np.array([ptscamRight.T[j,0],ptscamLeftInRight.T[j,0]]),np.array([ptscamRight.T[j,1],ptscamLeftInRight.T[j,1]]),np.array([ptscamRight.T[j,2],ptscamLeftInRight.T[j,2]]),'b')

    # plt.legend()
    # plt.show()
