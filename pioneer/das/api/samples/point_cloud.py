from pioneer.das.api.samples.sample import Sample

from typing import List, Optional, Tuple

import numpy as np
import warnings
warnings.simplefilter('once', DeprecationWarning)


class PointCloud(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super().__init__(index, datasource, virtual_raw, virtual_ts)

    def get_point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64) -> np.ndarray:
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
        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.get_field('t'), reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored
        
        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)

    @property
    def fields(self) -> Tuple[str]:
        return self.raw.dtype.names

    def get_field(self, field:str) -> Optional[np.ndarray]:
        if field in self.fields: 
            return self.raw[field]

    @property
    def distances(self):
        point_cloud = self.get_point_cloud()
        return (point_cloud[:,0]**2 + point_cloud[:,1]**2 + point_cloud[:,2]**2)**0.5

    @property
    def size(self) -> int:
        return self.raw.shape[0]



    ### Legacy section ###
    
    @property
    def timestamps(self):
        warnings.warn("PointCloud.timestamps is deprecated. Use PointCloud.get_field('t') instead.", DeprecationWarning)
        return self.get_field('t')
    
    @property
    def amplitudes(self):
        warnings.warn("PointCloud.amplitudes is deprecated. Use PointCloud.get_field('i') instead.", DeprecationWarning)
        return self.get_field('i').astype('f4')

    def get_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        warnings.warn("PointCloud.get_cloud() is deprecated. Use (PointCloud.get_point_cloud(), PointCloud.get_field('i'), np.arange(PointCloud.size)) instead.", DeprecationWarning)
        points = self.get_point_cloud(referential, ignore_orientation, undistort, reference_ts, dtype)
        return points, self.get_field('i'), np.arange(self.size)

    def point_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        warnings.warn("PointCloud.point_cloud() is deprecated. Use PointCloud.get_point_cloud() instead.", DeprecationWarning)
        return self.get_point_cloud(referential, ignore_orientation, undistort, reference_ts, dtype)
    
    def get_rgb_from_camera_projection(pcloud_ds:Sample, camera_ds:str, undistort:bool=False, return_mask:bool=False):
        from pioneer.das.api.datasources.virtual_datasources.rgb_cloud import get_rgb_from_camera_projection
        warnings.warn("PointCloud.get_rgb_from_camera_projection() is deprecated. Use from pioneer.das.api.datasources.virtual_datasources.rgb_cloud import get_rgb_from_camera_projection instead.", DeprecationWarning)
        return get_rgb_from_camera_projection(pcloud_ds, camera_ds, undistort, return_mask)