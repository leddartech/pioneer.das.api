from pioneer.das.api.samples.xyzit import XYZIT

import numpy as np

class EchoXYZIT(XYZIT):
    """Similar data structure than XYZIT sample. However, this sub-class should be 
        used instead if the sensor is a LCAx instance.
    """

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(EchoXYZIT, self).__init__(index, datasource, virtual_raw, virtual_ts)
    
    def point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        pts_Local = np.concatenate([self.raw['x'].reshape(-1, 1),
                            self.raw['y'].reshape(-1, 1),
                            self.raw['z'].reshape(-1, 1),], axis=1).astype(dtype)
        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.timestamps, reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored
        
        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)
