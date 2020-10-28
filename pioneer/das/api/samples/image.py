from pioneer.das.api.samples.sample import Sample

import cv2
import numpy as np

class Image(Sample):
    """RGB data from a single camera image"""

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
        ''' the undistorted new camera matrix'''
        if self._und_camera_matrix is not None:
            return self._und_camera_matrix
        
        v,h,_ = self.raw_image().shape
        self._und_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (h,v), 0.0, (h,v))
        return self._und_camera_matrix

    def project_pts(self, pts, mask_fov=False, output_mask=False, undistorted=False, margin=0):

        '''projects 3D points from camera referential to the image plane.

            Args:
                pts - (M,3)
                mask_fov - (True/False) If True the points outside fov are masked
                output_mask - (True/False) If True, the mask from mask_fov is returned
                undistorted - (True/False) If the image is undistorted
                margin (optionnal): margin (in pixels) outside the image unaffected by the fov mask
        '''

        R = T = np.zeros((3, 1))
        
        if undistorted or mask_fov or output_mask:
            A, k = self.und_camera_matrix, np.zeros((5,1))
            und_image_pts, _ = cv2.projectPoints(pts, R, T, A, k)
            und_image_pts = np.squeeze(und_image_pts)
            
        if not undistorted:
            A, k = self.camera_matrix, self.distortion_coeffs
            image_pts, _ = cv2.projectPoints(pts, R, T, A, k)
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
        
    def raw_image(self):
        if type(self.raw) == dict:
            return self.raw['image']
        return self.raw
    
    def undistort_image(self):
        return cv2.undistort(
            self.raw_image(), 
            cameraMatrix=self.camera_matrix, 
            distCoeffs=self.distortion_coeffs, 
            newCameraMatrix=self.und_camera_matrix,
        )

    def projection_mask(self, pts, projection, margin=0):
        v,h,_ = self.raw_image().shape
        mask = (pts[:,2] > 0)& \
            (projection[:,0] >= 0 - margin) & \
            (projection[:,0] <= h + margin) & \
            (projection[:,1] >= 0 - margin) & \
            (projection[:,1] <= v + margin)
        return mask
                                                    
