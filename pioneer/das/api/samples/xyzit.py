from pioneer.das.api.samples.sample import Sample

import numpy as np

class XYZIT(Sample):

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