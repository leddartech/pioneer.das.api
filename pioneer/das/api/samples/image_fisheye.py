from pioneer.common import mercator_projection
from pioneer.das.api.samples.image import Image

import cv2
import numpy as np

class ImageFisheye(Image):
    '''A derivation the Image sample, to be used with fisheye lenses.'''
    
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(ImageFisheye, self).__init__(index, datasource, virtual_raw, virtual_ts) 
        if not hasattr(self.datasource.sensor, 'mercator_projection'):
            self.datasource.sensor.mercator_projection = mercator_projection.MercatorProjection(self.camera_matrix, self.distortion_coeffs) 

    def project_pts(self, pts, mask_fov=False, output_mask=False, undistorted=False, margin=0):

        R = T = np.zeros((3, 1))
        
        if undistorted or mask_fov or output_mask:
            A, k = self.und_camera_matrix, np.zeros((5,1))
            und_image_pts = self.datasource.sensor.mercator_projection.project_pts(pts)
            und_image_pts = np.squeeze(und_image_pts)

        if not undistorted:
            A, k = self.camera_matrix, self.distortion_coeffs
            image_pts, _ = cv2.fisheye.projectPoints(pts.reshape((-1,1,3)), R, T, A, k)
            image_pts = np.squeeze(image_pts)
        else:
            image_pts = und_image_pts

        if mask_fov or output_mask:
            mask = self.projection_mask(pts, und_image_pts, margin)
            
        if mask_fov:
            image_pts = image_pts[mask]

        if output_mask:
            return image_pts, mask
        return image_pts

    def undistort_image(self):
        return self.datasource.sensor.mercator_projection.undistort(self.raw)

