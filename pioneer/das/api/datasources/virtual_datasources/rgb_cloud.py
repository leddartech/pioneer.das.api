from pioneer.common import platform
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Echo, PointCloud, Image

from typing import Any

import numpy as np


def get_rgb_from_camera_projection(pcloud_ds:PointCloud, camera_ds:str, undistort:bool=False, return_mask:bool=False):
    """Returns the rgb data for each point from its position in camera.
    
        Args:
            pcloud_ds: (Pointcloud) Pointcloud sample (ex: pf["ouster64_bfc_xyzit"][0])
            camera_ds: (str) name of the camera datasource (ex: 'flir_bbfc_flimg')
            undistort: (bool) if True, motion compensation is applied to the points before the projection (default is False)
            return_mask: (bool) if True, also returns the mask that only includes points inside the camera fov.

        Returns:
            rgb: A Nx3 array, where N is the number of points in the point cloud. RGB data is in the range [0,255]
            mask (optional): a Nx1 array of booleans. Values are True where points are inside the camera fov. False elsewhere.
    """

    image_sample:Image = pcloud_ds.datasource.sensor.pf[camera_ds].get_at_timestamp(pcloud_ds.timestamp)

    pcloud = pcloud_ds.get_point_cloud(referential=platform.extract_sensor_id(camera_ds), undistort=undistort)
    projection, mask = image_sample.project_pts(pcloud, mask_fov=True, output_mask=True)
    projection = projection.astype(int)

    rgb = np.zeros((pcloud.shape[0], 3))
    image = image_sample.get_image()
    rgb[mask,:] = image[projection[:,1], projection[:,0]]

    if return_mask:
        return rgb, mask
    return rgb


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
        pcloud = pcloud_sample.get_point_cloud(referential=self.camera_name, undistort=self.undistort)
        rgb_data, mask = get_rgb_from_camera_projection(pcloud_sample, self.original_image_datasource, undistort=self.undistort, return_mask=True)

        raw = np.empty(pcloud[mask].shape[0], dtype=self.dtype)
        if isinstance(pcloud_sample, PointCloud):
            raw['x'] = pcloud_sample.get_field('x')[mask]
            raw['y'] = pcloud_sample.get_field('y')[mask]
            raw['z'] = pcloud_sample.get_field('z')[mask]
            raw['i'] = pcloud_sample.get_field('i')[mask]
            raw['t'] = pcloud_sample.get_field('t')[mask]
        elif isinstance(pcloud_sample, Echo):
            # FIXME: Abstraction to remove this part
            pcloud = pcloud_sample.get_point_cloud(ignore_orientation=True)
            raw['x'] = pcloud[:,0][mask]
            raw['y'] = pcloud[:,1][mask]
            raw['z'] = pcloud[:,2][mask]
            raw['i'] = pcloud_sample.amplitudes[mask]
            raw['t'] = pcloud_sample.timestamps[mask]
        raw['r'] = rgb_data[mask,0]
        raw['g'] = rgb_data[mask,1]
        raw['b'] = rgb_data[mask,2]

        return PointCloud(key, self, raw, pcloud_sample.timestamp)