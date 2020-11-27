from pioneer.common.linalg import map_points, tf_inv
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit

from typing import Any

import copy
import numpy as np

try :
    import open3d
    OPEN3D_AVAILABLE = True
except:
    LoggingManager.instance().warning("Voxelisation will not work without installing open3d -> pip3 install open3d")
    OPEN3D_AVAILABLE = False

class VoxelMap(VirtualDatasource):
    """Merges multiple past points clouds in a single one, in the same present referential (mapping).
        Optionally, the merged point cloud can be voxelized (one averaged point per voxel, or volume unit).

        In order to work, the platform needs to have an egomotion_provider.
    """
    def __init__(self, reference_sensor:str, dependencies:list, memory:int=5, skip:int=1, voxel_size:float=0.0):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    The only element should be an echo or point cloud datasource (e.g. 'pixell_bfc_ech')
                memory (int): The number of past frames to merge.
                skip (int): If higher than 1, frames will be skipped. For example, with memory=10 and skip=2, 
                    the frames N=0,-2,-4,-6,-8,-10 will be merged. The present frame (N=0) is always included.
                voxel_size (float): If greater than 0, the merged point cloud will be voxelized.
        """
        self.has_rgb = 'xyzit-rgb' in dependencies[0]
        ds_type = 'xyzit-voxmap' if not self.has_rgb else 'xyzit-voxmap-rgb'
        super(VoxelMap, self).__init__(ds_type, dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_point_cloud_datasource = dependencies[0]
        self.memory = memory
        self.skip = skip
        self.voxel_size = voxel_size
        self.local_cache = None
        

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.original_point_cloud_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    @staticmethod
    def stack_point_cloud(stack, point_cloud):
        return np.vstack([stack, point_cloud])

    def clear_cache(self):
        self.local_cache = None

    def voxelize(self, pc):

        pc_open3d = open3d.geometry.PointCloud()
        pc_open3d.points = open3d.utility.Vector3dVector(pc[:,[0,1,2]])
        pc_open3d.colors = open3d.utility.Vector3dVector(pc[:,[3,3,3]])

        try:
            down_sampled = pc_open3d.voxel_down_sample(voxel_size=self.voxel_size)
        except:
            OPEN3D_AVAILABLE = False
            return pc_vox

        xyz_vox = np.asarray(down_sampled.points)
        int_vox = np.asarray(down_sampled.colors)[:,0]

        pc_vox = np.empty((xyz_vox.shape[0], pc.shape[1]))
        pc_vox[:,[0,1,2]] = xyz_vox
        pc_vox[:,3] = int_vox
        pc_vox[:,4] = pc[:,4].max()

        if self.has_rgb:
            pc_open3d.colors = open3d.utility.Vector3dVector(pc[:,[6,7,8]])
            down_sampled_rgb = pc_open3d.voxel_down_sample(voxel_size=self.voxel_size)
            pc_vox[:,[6,7,8]] = np.asarray(down_sampled_rgb.colors)[:,[0,1,2]]

        return pc_vox

    @property
    def _is_live(self):
        return self.sensor.platform.is_live()

    def __getitem__(self, key:Any):

        #TODO: if multiple point cloud datasources in dependencies, we could merge them.

        min_key = key - self.memory

        if not self._is_live:
            min_key = max([0, min_key])
        else:
            min_key = -min([-min_key, len(self.datasources[self.original_point_cloud_datasource])])

        samples = self.datasources[self.original_point_cloud_datasource][min_key:key+1]

        nb_features = 1 if not self.has_rgb else 4
        pc_map = np.empty((0,5+nb_features))

        cached_indices = []
        if self.local_cache is not None:
            pc_map = self.local_cache

            if not self._is_live:
                keep = np.where(
                    (pc_map[:,5] >= min_key) &\
                    (pc_map[:,5] <= key) &\
                    (pc_map[:,5] % self.skip == 0)
                )
            else:
                keep = np.where(
                    (pc_map[:,5] >= samples[0].raw['absolute_index']) &\
                    (pc_map[:,5] <= samples[-1].raw['absolute_index']) &\
                    (pc_map[:,5] % self.skip == 0)
                )
            pc_map = pc_map[keep]
            cached_indices = np.unique(pc_map[:,5])

        
        for sample in samples:

            if not self._is_live:
                index = sample.index
                if index % self.skip and index != key:
                    continue
            else:
                index = sample.raw['absolute_index']
                if index % self.skip and index != samples[-1].raw['absolute_index']:
                    continue

            if index in cached_indices:
                continue #don't re-add what is already cached

            pc = np.empty((sample.amplitudes.size,5+nb_features))
            pc[:,[0,1,2]] = sample.point_cloud(referential='world', undistort=False)
            pc[:,3] = sample.amplitudes
            pc[:,4] = sample.timestamp
            pc[:,5] = index

            if self.has_rgb:
                pc[:,6] = sample.raw['r']
                pc[:,7] = sample.raw['g']
                pc[:,8] = sample.raw['b']

            pc_map = self.stack_point_cloud(pc_map, pc)

        self.local_cache = copy.deepcopy(pc_map)

        if self.voxel_size > 0 and OPEN3D_AVAILABLE:
            pc_map = self.voxelize(pc_map)

        to_world = samples[-1].compute_transform('world')
        to_sensor = tf_inv(to_world)
        pc_map[:,[0,1,2]] = map_points(to_sensor, pc_map[:,[0,1,2]])

        #package in das format
        dtype = datasource_xyzit() if not self.has_rgb else sample.raw.dtype
        raw = np.empty((pc_map.shape[0]), dtype=dtype)
        raw['x'] = pc_map[:,0]
        raw['y'] = pc_map[:,1]
        raw['z'] = pc_map[:,2]
        raw['i'] = pc_map[:,3]
        raw['t'] = pc_map[:,4]

        if self.has_rgb:
            raw['r'] = pc_map[:,6]
            raw['g'] = pc_map[:,7]
            raw['b'] = pc_map[:,8]

        sample_object = self.sensor.factories['xyzit'][0]

        return sample_object(index=key, datasource=self, virtual_raw=raw, virtual_ts=samples[-1].timestamp)

