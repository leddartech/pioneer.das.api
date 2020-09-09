from numpy import linalg as LA
from scipy.linalg import expm

import math
import matplotlib.pyplot as plt
import numpy as np

#Basic functions and stuffs:
def fnorm(A):
    """Frobenius norm"""
    return LA.norm(A, ord='fro')

def vex(A):
    """Vector-like extraction
    """
    return np.array([A[2,1], A[0,2], A[1,0]], dtype=np.float)

def mat_log(A):
    """ Log of a matrix
    """
    B = np.copy(0.5*(A-A.T))
    b = vex(B)
    b_norm = LA.norm(b)

    if (B==0).all():
        return np.zeros_like(A)
    else:
        return np.arcsin(b_norm)/b_norm*B
        
def mat_exp(A):
    """Expo of a matrix, using Pade diff eqs
    """
    return expm(A)

def d_SO3_ang2(A,B):
    """Compute the angular distance between two rotations (ie angle of rotation of AB.T)

    """
    return 2**(-0.5)*fnorm(mat_log(np.dot(A,B.T)))

def angular_L2_mean(A, epsilon=1e-5, n_iters=100):
    """Algo1 of L2 mean of elements in SO(3).  

        Args:
            A - list of matrix =[A0, A1, A2, ...]
            epsilon - tolerance >0
            n_iters - the max number of iterations>0
        
        Return:
            Rotation matrix R
    """
    R = np.copy(A[0])
    n = len(A)
    for mu in range(n_iters):
        B = 1/n*np.sum(np.vstack([ [mat_log(np.dot(R.T,x))] for x in A]), axis=0)
        if fnorm(B)<epsilon:
            return R
        else:
            R = np.copy(np.dot(R,mat_exp(B)))

    print('Max number of iterations...')
    return R

if __name__ == '__main__':
    #testing

    from scipy.stats import special_ortho_group

    #Generate few random orthogonal matrices
    x = special_ortho_group.rvs(3)
    y = special_ortho_group.rvs(3)
    z = special_ortho_group.rvs(3)
    w = special_ortho_group.rvs(3)
    A = [x,y,z,w]

    R = angular_L2_mean(A, epsilon=1e-5, n_iters=100)
    print('Resulting matrix: ', R)
    print('Unit test:', np.dot(R, R.T))



    


