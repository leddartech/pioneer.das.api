from pioneer.common import platform
from pioneer.das.api.samples.sample import Sample

import numpy as np

class XYZIT(Sample):
    """Point cloud provided by a mechanical lidar sensor.
        For each data point, contains (x,y,z) coordinates, the intensity (i) and a timestamp (t).
    """

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(XYZIT, self).__init__(index, datasource, virtual_raw, virtual_ts)
    
    @property
    def timestamps(self):
        return self.raw['t']
    
    @property
    def amplitudes(self):
        return self.raw['i'].astype('f4')

    @property
    def distances(self):
        return (self.raw['x']**2 + self.raw['y']**2 + self.raw['z']**2)**0.5

    def get_cloud( self
                 , referential = None
                 , ignore_orientation=False
                 , undistort=False
                 , reference_ts=-1
                 , dtype = np.float64):

        points = self.point_cloud(referential, ignore_orientation, undistort, reference_ts, dtype)

        return points, self.amplitudes, np.arange(points.shape[0])

    def point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        """Compute a 3D point cloud from raw data
        
        Args:
            referential: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            undistort: Apply motion compensation to 3d points.
            reference_ts:  (only used if referential == 'world' and/or undistort == True), 
                           refer to compute_transform()'s documentation
            dtype: the output numpy data type
        """
        pts_Local = np.concatenate([self.raw['x'].reshape(-1, 1),
                            self.raw['y'].reshape(-1, 1),
                            self.raw['z'].reshape(-1, 1),], axis=1).astype(dtype)
        pts_Local = self.datasource.sensor.get_corrected_cloud(self.timestamp, pts_Local, dtype)
        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.timestamps, reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored
        
        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)


    def get_rgb_from_camera_projection(self, camera:str, undistort:bool=False, return_mask:bool=False):
        """Returns the rgb data for each point from its position in camera.
        
            Args:
                camera: (str) name of the camera datasource (ex: 'flir_bbfc_flimg')
                undistort: (bool) if True, motion compensation is applied to the points before the projection (default is False)
                return_mask: (bool) if True, also returns the mask that only includes points inside the camera fov.

            Returns:
                rgb: A Nx3 array, where N is the number of points in the point cloud. RGB data is in the range [0,255]
                mask (optional): a Nx1 array of booleans. Values are True where points are inside the camera fov. False elsewhere.
        """

        image_sample = self.datasource.sensor.pf[camera].get_at_timestamp(self.timestamp)

        pcloud = self.point_cloud(referential=platform.extract_sensor_id(camera), undistort=undistort)
        projection, mask = image_sample.project_pts(pcloud, mask_fov=True, output_mask=True)
        projection = projection.astype(int)

        rgb = np.zeros((pcloud.shape[0], 3))
        image = image_sample.raw_image()
        rgb[mask,:] = image[projection[:,1], projection[:,0]]

        if return_mask:
            return rgb, mask
        return rgb
