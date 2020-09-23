from pyquaternion import Quaternion

import math
import numpy as np

# Splitting the transformation in translation and rotation is a good start.
# Averaging the translation is trivial.
# Averaging the rotation is not that easy. Most approaches will use quaternions. So you need to transform the rotation matrix to a quaternion.
# The easiest way to approximate the average is a linear blending, followed by renormalization of the quaternion:
# The reason for that is that the combination of two rotations is not performed by adding the quaternions, but by multiplying them.
# If we convert quaternions to a logarithmic space, we can use a simple linear blend (because multiplication will become additions).
# Then transform the quaternion back to the original space. This is the idea of the Spherical Average (Buss 2001).

class Rotation_Average(object):

    def Translation_Part(self,matrix):

        translation = np.zeros((3, 1))
        for i in range(0, 3):
            translation[i] = matrix[i][3]

        return translation

    def Rotation_Part(self,matrix):

        rotation = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0,3):
                rotation[i][j] = matrix[i][j]

        return rotation

    def AxisAngle_Quaternion(self,Axis,angle):

        quaternion = Quaternion(axis =(Axis[0], Axis[1], Axis[2]),radians= angle)

        return quaternion

    def log_Quaternion(self,quaternion):

        log_quaternion = Quaternion.log(quaternion)

        return log_quaternion

    def Quaternion_R(self,log_quaternion):

        q_exp = Quaternion.exp(log_quaternion)
        unit = q_exp.normalised
        R = unit.rotation_matrix

        return R

    def R_to_axis_angle(self,matrix):
        """Convert the rotation matrix into the axis-angle notation.
        Conversion equations
        ====================
        From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
            x = Qzy-Qyz
            y = Qxz-Qzx
            z = Qyx-Qxy
            r = hypot(x,hypot(y,z))
            t = Qxx+Qyy+Qzz
            theta = atan2(r,t-1)
        @param matrix:  The 3x3 rotation matrix to update.
        @type matrix:   3x3 numpy array
        @return:    The 3D rotation axis and angle.
        @rtype:     numpy 3D rank-1 array, float
        """

        # Axes.
        axis = np.zeros(3, np.float64)
        axis[0] = matrix[2, 1] - matrix[1, 2]
        axis[1] = matrix[0, 2] - matrix[2, 0]
        axis[2] = matrix[1, 0] - matrix[0, 1]

        # Angle.
        r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
        t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        theta = math.atan2(r, t - 1)

        # Normalise the axis.
        axis = axis / r

        # Return the data.
        return axis, theta

    def R_to_Euler(self,matrix):
        """ Decomposing rotation matrix to 3 Euler anglesConvert the rotation matrix into the axis-angle notation.

        thetax = atan2(r32,r33)
        thetay = atan2(-r31,sqrt(r32² + r33²))
        thetaz = atan2(r21,r11)
        """

        # Angle.
        thetax = math.atan2(matrix[2][1],matrix[2][2])
        thetay = math.atan2(-matrix[2][0],np.sqrt(np.square(matrix[2][1]) + np.square(matrix[2][2])))
        thetaz = math.atan2(matrix[1][0],matrix[0][0])

        # Corresponding matrices
        Rx = np.eye(3)
        Rx[1][1] = math.cos(thetax)
        Rx[1][2] = -math.sin(thetax)
        Rx[2][1] = math.sin(thetax)
        Rx[2][2] = math.cos(thetax)
        Ry = np.eye(3)
        Ry[0][0] = math.cos(thetay)
        Ry[0][2] = math.sin(thetay)
        Ry[2][0] = -math.sin(thetay)
        Ry[2][2] =  math.cos(thetay)
        Rz = np.eye(3)
        Rz[0][0] = math.cos(thetaz)
        Rz[1][0] = math.sin(thetaz)
        Rz[1][0] = -math.sin(thetaz)
        Rz[1][1] = math.cos(thetaz)

        # Return the data.
        return Rx,Ry,Rz
    
    def R_to_Angle_degree(self,matrix):
        """ Decomposing rotation matrix to 3 Euler anglesConvert the rotation matrix into the axis-angle notation.

        thetax = atan2(r32,r33)
        thetay = atan2(-r31,sqrt(r32² + r33²))
        thetaz = atan2(r21,r11)
        """

        # Angle.
        thetax = math.atan2(matrix[2][1],matrix[2][2])
        thetay = math.atan2(-matrix[2][0],np.sqrt(np.square(matrix[2][1]) + np.square(matrix[2][2])))
        thetaz = math.atan2(matrix[1][0],matrix[0][0])

        return np.rad2deg(thetax),np.rad2deg(thetay),np.rad2deg(thetaz)
    
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self,theta) :
        
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
            
            
                        
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
                    
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
                        
                        
        R = np.dot(R_z, np.dot( R_y, R_x ))
    
        return R