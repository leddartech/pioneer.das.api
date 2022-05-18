from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit
from pioneer.das.api.samples.point_cloud import PointCloud

from typing import Any

import numpy as np


class PointCloudFusion(VirtualDatasource):
    """Merges multiple points clouds in a single one"""

    def __init__(self, reference_sensor:str, dependencies:list):
        """Constructor
            Args:
                reference_sensor (str): The name of the reference sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the point cloud datasources to fuse. 
        """
        ds_type = 'xyzit-fused'
        super(PointCloudFusion, self).__init__(ds_type, dependencies, None)
        self.reference_sensor = reference_sensor  
        self.reference_datasource = dependencies[0]  

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.reference_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    @staticmethod
    def stack_point_cloud(stack, point_cloud):
        return np.vstack([stack, point_cloud])

    def clear_cache(self):
        self.local_cache = None

    def __getitem__(self, key:Any):

        pcloud_fused = np.empty((0,5))

        for ds_name in self.dependencies:
            sample = self.datasources[ds_name][key]
            xyz = sample.get_point_cloud(referential=self.reference_sensor)

            if self.sensor.orientation is not None:
                xyz = xyz @ self.sensor.orientation

            pcloud = np.empty((xyz.shape[0],5))
            pcloud[:,[0,1,2]] = xyz
            pcloud[:,3] = sample.get_field('i')
            pcloud[:,4] = sample.get_field('t')

            pcloud_fused = self.stack_point_cloud(pcloud_fused, pcloud)

        #package in das format
        dtype = datasource_xyzit()
        raw = np.empty((pcloud_fused.shape[0]), dtype=dtype)
        raw['x'] = pcloud_fused[:,0]
        raw['y'] = pcloud_fused[:,1]
        raw['z'] = pcloud_fused[:,2]
        raw['i'] = pcloud_fused[:,3]
        raw['t'] = pcloud_fused[:,4]

        ts = self.datasources[self.dependencies[0]][key].timestamp
        return PointCloud(index=key, datasource=self, virtual_raw=raw, virtual_ts=ts)

