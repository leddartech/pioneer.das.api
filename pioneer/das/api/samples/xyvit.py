from pioneer.das.api.samples.xyzit import XYZIT

import numpy as np

class XYVIT(XYZIT):
    """Point cloud in bird eye view plane provided by a radar sensor.
        For each data point, contains (x,y) coordinates, the relative radial velocity (v),
        the intensity (i) and a timestamp (t).
    """

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(XYVIT, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        """Compute a 2D BEV point cloud from raw data"""

        pts_Local = np.concatenate([self.raw['x'].reshape(-1, 1),
                            self.raw['y'].reshape(-1, 1),
                            np.zeros_like(self.raw['x']).reshape(-1, 1),], axis=1).astype(dtype)
        pts_Local = self.datasource.sensor.get_corrected_cloud(self.timestamp, pts_Local, dtype)
        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.timestamps, reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored
        
        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)
