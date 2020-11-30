from pioneer.common import platform
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit_float_intensity
from pioneer.das.api.samples import Echo, XYZIT

from typing import Any

import numpy as np

class RGBCloud(VirtualDatasource):
    """Point cloud with RGB data from the camera projection"""

    def __init__(self, reference_sensor:str, dependencies:list, undistort:bool=False):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    The first element should be a point cloud datasource (e.g. 'pixell_bfc_ech')
                    The second element should be a camera image datasource (e.g. 'flir_bfc_img')
                undistort (bool): if True, motion compensentation is applied to the pcloud before the camera projection. 
                    Doesn't affect the point positions, only the rgb data.
        """
        super(RGBCloud, self).__init__(f'xyzit-rgb', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_pcloud_datasource = dependencies[0]
        self.original_image_datasource = dependencies[1]
        self.camera_name = platform.extract_sensor_id(dependencies[1])
        self.undistort = undistort

        self.dtype = np.dtype([('x','f4'),('y','f4'),('z','f4'),('i','u2'),('t','u8'),('r','u8'),('g','u8'),('b','u8')])

    def __getitem__(self, key:Any):

        if isinstance(key, slice):
            return self[platform.slice_to_range(key, len(self))]
        if isinstance(key, range):
            return [self[index] for index in key]

        pcloud_sample = self.datasources[self.original_pcloud_datasource][key]
        pcloud = pcloud_sample.point_cloud(referential=self.camera_name, undistort=self.undistort)
        rgb_data, mask = pcloud_sample.get_rgb_from_camera_projection(self.original_image_datasource, undistort=self.undistort, return_mask=True)

        raw = np.empty(pcloud[mask].shape[0], dtype=self.dtype)
        if isinstance(pcloud_sample, XYZIT):
            raw['x'] = pcloud_sample.raw['x'][mask]
            raw['y'] = pcloud_sample.raw['y'][mask]
            raw['z'] = pcloud_sample.raw['z'][mask]
            raw['i'] = pcloud_sample.raw['i'][mask]
            raw['t'] = pcloud_sample.raw['t'][mask]
        elif isinstance(pcloud_sample, Echo):
            pcloud = pcloud_sample.point_cloud(ignore_orientation=True)
            raw['x'] = pcloud[:,0][mask]
            raw['y'] = pcloud[:,1][mask]
            raw['z'] = pcloud[:,2][mask]
            raw['i'] = pcloud_sample.amplitudes[mask]
            raw['t'] = pcloud_sample.timestamps[mask]
        raw['r'] = rgb_data[mask,0]
        raw['g'] = rgb_data[mask,1]
        raw['b'] = rgb_data[mask,2]

        sample_object = self.sensor.factories['xyzit'][0]
        return sample_object(key, self, raw, pcloud_sample.timestamp)