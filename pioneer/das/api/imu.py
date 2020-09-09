import math 
import numpy as np
import transforms3d 
import utm

def axangle4_to_3(ax):
    normal = ax[0]
    angle = ax[1]
    return normal*angle


def axangle3_to_4(ax):
    angle = np.linalg.norm(ax)
    if(angle != 0):
        return (ax/angle, angle)
    else:
        return ([0.0, 0.0, 1.0], 0)

def x_to_mat(x):
    x0 = np.eye(4, dtype=np.float64)
    t = x[0:3]
    x0[0:3, 3] = t
    ax = x[3:6]
    axx, angle = axangle3_to_4(ax)
    # print('axx, angle', axx, angle)
    x0[0:3, 0:3] = transforms3d.axangles.axangle2mat(axx, angle)
    return x0


def x_quaternion_to_mat(x):
    x0 = np.eye(4, dtype=np.float64)
    t = x[0:3]
    x0[0:3, 3] = t
    ax = x[3:7]
    axx, angle = axangle3_to_4(ax)
    # print('axx, angle', axx, angle)
    x0[0:3, 0:3] = transforms3d.axangles.axangle2mat(axx, angle)
    return x0


def x_quaternion_to_mat(x):
    x0 = np.eye(4, dtype=np.float64)
    t = x[0:3]
    x0[0:3, 3] = t
    q = x[3:7]
    # print('axx, angle', axx, angle)
    x0[0:3, 0:3] = transforms3d.quaternions.quat2mat(q)
    return x0


def axangle_X0_init(mat):

    t = mat[0:3, 3]
    r = mat[0:3, 0:3]

    ax = transforms3d.axangles.mat2axangle(r)
    ax3 = axangle4_to_3(ax)
    x0 = np.zeros((6), dtype=np.float64)
    x0[0:3] = t
    x0[3:6] = ax3
    return x0

def quaternion_X0_init(mat):

    t = mat[0:3, 3]
    r = mat[0:3, 0:3]
    q = transforms3d.quaternions.mat2quat(r)
    x0 = np.zeros((7), dtype=np.float64)
    x0[0:3] = t
    x0[3:7] = q
    return x0


def state_2_affine(x):
    tmp = np.eye(4, dtype=np.float64)
    t = x[0:3]
    ax = x[3:6]
    ax, angle = axangle3_to_4(ax)
    # print('axx, angle2', ax, angle)
    R = transforms3d.axangles.axangle2mat(ax, angle)
    tmp[0:3, 0:3] = R
    tmp[0:3, 3] = t
    return tmp

def get_euler_angleZYX(R):
    angle = np.zeros(3, dtype=np.float64)

    angle[1] = math.asin(R[2, 0])  # math.asin(R[0, 2])
    # math.acos(R[0, 0] / math.cos(angle[1]))
    angle[2] = math.atan2(-R[1, 0], R[0, 0])
    # math.acos(R[2, 2] / math.cos(angle[1]))
    angle[0] = math.atan2(-R[2, 1], R[2, 2])
    return -angle

def pose_to_transf(pose):
    Tr = np.eye(4)
    Tr[0:3, 3] = pose[0:3]
    roll = pose[3]
    pitch = pose[4]
    yaw = pose[5]

    Rx = np.matrix([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.matrix([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.matrix([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    Tr[0:3, 0:3] = Rz * Ry * Rx # pout y a un bug ici --> c'est sur c'est corrige! vérifié dans la doc de la landins
    # --> attention le get euler angle doit aller avec l'ordre de multiplication des matrices
    #FIXME: use transforms3d.euler2mat(roll, pitch, yaw, 'sxyz')
    return Tr

def imu_to_pose(imuVector):
    # yaw: cap magnétique sens horaire, orienté nord (0deg == nord), repere utm: x->est, y->nord, z->haut
    
    x, y, zoneNumber, zoneLetter = utm.from_latlon(imuVector[0], imuVector[1])
    pose = np.array([x, y, imuVector[2], (imuVector[3]), (imuVector[4]), (-imuVector[5] + math.pi / 2)]) #FIXME: why a negative on imuVector[5] and another for 'pitch' in get_imu_pose()
    return pose 

def get_imu_pose(nav, euler):
    # centrale en NED, 
    return np.array([nav['latitude'],nav['longitude'],nav['altitude'],euler['roll'], -euler['pitch'], euler['yaw']]) #FIXME see imu_to_pose(), also, why two functions?

def inverse_transform(t):
    tr = np.eye(4)
    rr = t[0:3, 0:3].transpose()
    tt = t[0:3, 3]
    tr[0:3, 0:3] = rr
    tr[0:3, 3] = -rr.dot(tt)
    return tr

def print_transf_result(Tr):

    x = axangle_X0_init(Tr)
    print('translations :{}, norm:{}'.format(x[0:3], np.linalg.norm(x[0:3])))
    ax = axangle3_to_4(x[3:6])
    print('Ax Angle vector:{}, angle (deg):{}'.format(ax[0], np.rad2deg(ax[1])))
    print('Euler Angle (deg):{}'.format(np.rad2deg(get_euler_angleZYX(Tr))))