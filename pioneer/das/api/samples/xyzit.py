from pioneer.das.api.samples.point_cloud import PointCloud

import numpy as np

class XYZIT(PointCloud):
    """Point cloud provided by a mechanical lidar sensor.
        For each data point, contains (x,y,z) coordinates, the intensity (i) and a timestamp (t).
    """

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super().__init__(index, datasource, virtual_raw, virtual_ts)
    
    def get_point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        """Compute a 3D point cloud from raw data
        
        Args:
            referential: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            undistort: Apply motion compensation to 3d points.
            reference_ts:  (only used if referential == 'world' and/or undistort == True), 
                           refer to compute_transform()'s documentation
            dtype: the output numpy data type
        """
        pts_Local = np.concatenate([self.get_field('x').reshape(-1, 1),
                            self.get_field('y').reshape(-1, 1),
                            self.get_field('z').reshape(-1, 1),], axis=1).astype(dtype)

        # Legacy stuff with Ouster64's temperature compensation.
        if hasattr(self.datasource.sensor, "get_corrected_cloud"):
            pts_Local = self.datasource.sensor.get_corrected_cloud(self.timestamp, pts_Local, dtype)

        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.get_field('t'), reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored
        
        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)