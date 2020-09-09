from pioneer.common import mercator_projection
from pioneer.das.api.samples.image import Image

import cv2
import numpy as np

class ImageFisheye(Image):

    '''A derivation of a standard Flir Image, whenever the lens used is of type fisheye (from immervision)
    '''
    
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(ImageFisheye, self).__init__(index, datasource, virtual_raw, virtual_ts) 
        if not hasattr(self.datasource.sensor, 'mercator_projection'):
            self.datasource.sensor.mercator_projection = mercator_projection.MercatorProjection(self.camera_matrix, self.distortion_coeffs) 

    def project_pts(self, pts, output_mask=None, undistorted=False):
        if undistorted:
            return self.datasource.sensor.mercator_projection.project_pts(pts)
        
        R = T = np.zeros((3, 1))
        A, k = self.camera_matrix, self.distortion_coeffs
        pts = pts.reshape((-1,1,3))
        image_pts, _ = cv2.fisheye.projectPoints(pts, R, T, A, k)

        return np.squeeze(image_pts)

    def undistort_image(self):
        return self.datasource.sensor.mercator_projection.undistort(self.raw)

