from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.samples import Image, ImageFisheye
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.interpolators import nearest_interpolator

import numpy as np

class Camera(Sensor):
    """Camera sensor, expects 'img' datasource, intrinsics matrix and distortion coefficients"""

    def __init__(self, name, platform):
        super(Camera, self).__init__(name
                                   , platform
                                   , { 'img': (Image, nearest_interpolator),
                                       'flimg': (ImageFisheye, nearest_interpolator)
                                        })

    @property
    def camera_matrix(self):
        """np.ndarray: the 3x3 intrinsics matrix
        
        See also: https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        """ 
        try:
            matrix = self.intrinsics['matrix']
        except:
            LoggingManager.instance().warning(f'Intrinsic matrix of {self.name} not found. Trying to make a generic one.')
            h = self.yml['configurations']['img']['Width']
            v = self.yml['configurations']['img']['Height']
            h_fov = self.yml['configurations']['img']['h_fov']
            matrix = np.identity(3)
            matrix[0,2] = h/2
            matrix[1,2] = v/2
            matrix[0,0] = matrix[1,1] = h/(2*np.tan(h_fov * np.pi / 360.0))
        return matrix

    @property
    def distortion_coeffs(self):
        """np.ndarray: Nx1 distortion coefficient, refer to opencv documentation 
        
        See also: https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        """
        try:
            k = self.intrinsics['distortion']
        except:
            LoggingManager.instance().warning(f'Distortion coeffients not found for {self.name}.')
            k = np.zeros((5,1))
        return k

    def load_intrinsics(self, intrinsics_config):
        super(Camera, self).load_intrinsics(intrinsics_config)
        if self.intrinsics is not None:
            assert 'matrix' in self.intrinsics
            # previous calibration had a typo in the distortion coeffs
            if (not 'distortion' in self.intrinsics and
                'distorsion' in self.intrinsics):
                self.intrinsics['distortion'] = self.intrinsics['distorsion']
            assert 'distortion' in self.intrinsics
