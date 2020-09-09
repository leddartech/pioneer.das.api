from pioneer.common.linalg import tf_inv, map_points
from pioneer.common.logging_manager import LoggingManager
from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit

from typing import Any

import numpy as np
try :
    import open3d
    OPEN3D_AVAILABLE = True
except:
    LoggingManager.instance().warning("Voxelisation will not work without installing open3d -> pip3 install open3d")
    OPEN3D_AVAILABLE = False
import warnings

class VoxelMap(VirtualDatasource):

    def __init__(self, ds_type, original_datasource, sensor=None, memory:int=10, voxel_size:float=0.1):
        super(VoxelMap, self).__init__(ds_type, [original_datasource], None)
        self.ds_type = ds_type
        self.original_datasource = original_datasource
        self.sensor = sensor
        self.memory = memory
        self.voxel_size = voxel_size
        self.local_cache = {'keys': None}


    @staticmethod
    def add_all_combinations_to_platform(pf:'Platform', memory:int=10, voxel_size:float=0.1) -> list:
        try:
            all_3d_datasources = pf.expand_wildcards(["*_ech","*_xyzit","*_rad"])
            virtual_ds_list = []

            for original_datasource_full_name in all_3d_datasources:
                original_datasource, pos, ds_type = parse_datasource_name(original_datasource_full_name)
                sensor = pf[f"{original_datasource}_{pos}"]
                virtual_ds_type = "xyzit-voxmap"

                try:
                    vds = VoxelMap(
                        ds_type = virtual_ds_type,
                        original_datasource = original_datasource_full_name,
                        sensor = sensor,
                        memory = memory,
                        voxel_size = voxel_size,
                    )
                    sensor.add_virtual_datasource(vds)
                    virtual_ds_list.append(f"{original_datasource}_{pos}_{virtual_ds_type}")
                except Exception as e:
                    print(e)
                    print(f"vitual datasource {original_datasource}_{pos}_{virtual_ds_type} was not added")
                
            return virtual_ds_list
        except Exception as e:
            print(e)
            print("Issue during try to add virtual datasources VoxelMap.")


    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.original_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]


    @staticmethod
    def stack_point_cloud(stack, point_cloud):
        return np.vstack([stack, point_cloud])


    def voxelize(self, pc):

        pc_open3d = open3d.geometry.PointCloud()
        pc_open3d.points = open3d.utility.Vector3dVector(pc[:,[0,1,2]])
        pc_open3d.colors = open3d.utility.Vector3dVector(pc[:,[3,3,3]])

        down_sampled = pc_open3d.voxel_down_sample(voxel_size=self.voxel_size)
        xyz_vox = np.asarray(down_sampled.points)
        int_vox = np.asarray(down_sampled.colors)[:,0]

        pc_vox = np.empty((xyz_vox.shape[0],5))
        pc_vox[:,[0,1,2]] = xyz_vox
        pc_vox[:,3] = int_vox
        pc_vox[:,4] = pc[:,4].max()

        return pc_vox


    def __getitem__(self, key:Any):

        pc_map = np.empty((0,5))

        min_key = max([0, key-self.memory])

        if self.local_cache['keys'] == f'{min_key}:{key+1}':
            pc_map = self.local_cache['pc_map']
        else:
            samples = self.datasources[self.original_datasource][min_key:key+1]
            for sample in samples:
                pc = np.empty((sample.amplitudes.size,5))
                pc[:,[0,1,2]] = sample.point_cloud(referential='world', undistort=True)
                pc[:,3] = sample.amplitudes
                pc[:,4] = sample.timestamp
                pc_map = self.stack_point_cloud(pc_map, pc)

            to_world = samples[-1].compute_transform('world')
            to_sensor = tf_inv(to_world)
            pc_map[:,[0,1,2]] = map_points(to_sensor, pc_map[:,[0,1,2]])

            self.local_cache = {
                'keys':f'{min_key}:{key+1}', 
                'pc_map': pc_map, 
                'timestamp': samples[-1].timestamp
            }

        if self.voxel_size is not None and OPEN3D_AVAILABLE:
            pc_map = self.voxelize(pc_map)
        
        raw = np.empty((pc_map.shape[0]),dtype=datasource_xyzit())
        raw['x'] = pc_map[:,0]
        raw['y'] = pc_map[:,1]
        raw['z'] = pc_map[:,2]
        raw['i'] = pc_map[:,3]
        raw['t'] = pc_map[:,4]

        sample_object = self.sensor.factories['xyzit'][0]

        return sample_object(index=key, datasource=self, virtual_raw=raw, virtual_ts=self.local_cache['timestamp'])

