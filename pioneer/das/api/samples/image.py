from pioneer.das.api.samples.sample import Sample

import cv2
import numpy as np

class Image(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Image, self).__init__(index, datasource, virtual_raw, virtual_ts)

        self._und_camera_matrix = None

    @property
    def camera_matrix(self):
        matrix = self.datasource.sensor.camera_matrix
        return matrix

    @property
    def distortion_coeffs(self):
        k = self.datasource.sensor.distortion_coeffs
        return k
    
    @property
    def und_camera_matrix(self):
        ''' the undistorted new camera matrix
        '''
        if self._und_camera_matrix is not None:
            return self._und_camera_matrix
        
        v,h,_ = self.raw.shape
        self._und_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, 
                                                    self.distortion_coeffs, 
                                                    (h,v), 
                                                    0.0, 
                                                    (h,v))
        return self._und_camera_matrix

    def project_pts(self, pts, output_mask=None, undistorted=False):
        R = T = np.zeros((3, 1))
        if undistorted:
            A, k = self.und_camera_matrix, np.zeros((5,1))
        else:
            A, k = self.camera_matrix, self.distortion_coeffs
        image_pts, _ = cv2.projectPoints(pts, R, T, A, k)
        return np.squeeze(image_pts)
        
    def raw_image(self):
        return self.raw
    
    def undistort_image(self):
        image = self.raw_image()
        undistorted = cv2.undistort(image, cameraMatrix=self.camera_matrix, 
                                distCoeffs=self.distortion_coeffs, newCameraMatrix=self.und_camera_matrix)
        
        return undistorted
                                                    
