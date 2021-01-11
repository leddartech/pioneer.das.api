from pioneer.common import banks, clouds, constants
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.sensors.lcax import LCAx

from typing import Union

import numpy as np


class Pixell(LCAx):

    def __init__(self, name, platform):
        super(Pixell, self).__init__(name, platform)
        self.orientation = np.array([[ 0, 0, 1],
                                     [-1, 0, 0],
                                     [ 0, 1, 0]], 'f8')
        self.adc_freq = 100e6
        self.specs = {'v':8, 'h':96, 'v_fov':16.073, 'h_fov':180.604}
        self.head_positions = constants.PIXELL_HEAD_POSITIONS
        self.head_position_per_channel = self.get_head_position_per_channel()
        self.calibrated_angles = None

    def load_intrinsics(self, intrinsics_config: Union[str, dict]):
        '''Basic intrinsics are brougth back under Sensor.
        '''
        super(Pixell, self).load_intrinsics(intrinsics_config)
        try:
            cfg = self['cfg'][0].raw
            self.azimut = np.deg2rad(cfg['calibration']['ID_CHANNEL_ANGLE_AZIMUT'])
            self.elevation = np.deg2rad(cfg['calibration']['ID_CHANNEL_ANGLE_ELEVATION'])
            self.calibrated_angles = np.c_[self.elevation, self.azimut].astype(np.float)
            _time_base_delays = cfg['calibration']['ID_TIMEBASE_DELAY']
            self.time_base_delays = {
                'low': _time_base_delays[:self.specs['v']*self.specs['h']][constants.ID_FROM_LCAS_CHAN_TO_SENSOR_CHAN],
                'high':_time_base_delays[self.specs['v']*self.specs['h']:][constants.ID_FROM_LCAS_CHAN_TO_SENSOR_CHAN],
            }
        except:
            pass

    def cache(self, specs):
        hashable = frozenset([(k,specs[k]) for k in ['v', 'h', 'v_fov', 'h_fov']])
        if not hashable in self._cache:
            cache = super(Pixell, self).cache(specs)
            cache['lcas_offsets'] = self.head_position_per_channel
            if self.calibrated_angles is not None:
                cache['angles'] = self.calibrated_angles
                cache['quad_angles'] = clouds.custom_quad_angles(specs, cache['angles'])
            else:
                cache['quad_angles'] = clouds.quad_angles(specs)
            self._cache[hashable] = cache

        return self._cache[hashable]

    def get_trace_smoothing_kernel(self):
        return constants.PIXELL_FILTERS_KERNEL

    def get_corrected_cloud(self, timestamp, cache, type, indices, distances, amplitudes=None, dtype=np.float64):
        
        if self.calibrated_angles is not None:
            type_pts, type_amplitudes, type_indices = self.get_cloud(cache, type, indices, distances, amplitudes, dtype)

            if type == 'point_cloud':
                return type_pts
            elif type == 'quad_cloud':
                return type_pts, type_amplitudes, type_indices
            else:
                raise KeyError('Can not compute the point cloud projection')
        else:
            return super(Pixell, self).get_corrected_cloud(timestamp, cache, type, indices, distances, amplitudes, dtype)

    def get_cloud(self, cache, type, indices, distances, amplitudes=None, dtype=np.float64):
        '''Get cloud data (type_pts, type_amplitudes, type_indices) from a sensor.'''

        if type == 'point_cloud':
            x,y,z = self.xyz_from_heads(distances, cache['angles'][indices], cache['lcas_offsets'][indices])
            pts = np.stack(( x,  y, - z), axis=-1)

            pts = pts @ self.orientation
            return pts, amplitudes, indices
        
        elif type == 'quad_cloud':
            distances_, elevation, azimut = [],[],[]
            m_ = cache['corrected_specs']['v'] * cache['corrected_specs']['h']
            for i in range(4):
                elevation_, azimut_ = self.angles_from_heads(distances, cache['quad_angles'][indices + i * m_], cache['lcas_offsets'][indices])
                distances_.extend(distances)
                elevation.extend(elevation_)
                azimut.extend(azimut_)

            distances_ = np.array(distances_, dtype=np.float)
            elevation = np.array(elevation, dtype=np.float)
            azimut = np.array(azimut, dtype=np.float)
            y,z,x = clouds.direction_spherical(elevation, azimut)
            pts = np.stack((distances_ * x, distances_ * y, -distances_ * z), axis=-1)
            pts = pts @ self.orientation
            quad_amplitudes = clouds.quad_stack(amplitudes)
            quad_indices = clouds.generate_quads_indices(indices.shape[0], np.uint32).flatten()

            return pts, quad_amplitudes, quad_indices

    @staticmethod
    def angles_from_heads(distances, angles, head_positions):
        distances_ = distances-np.cos(angles[:,0]) * (head_positions[:,1] * np.sin(angles[:,1]) + head_positions[:,0] * np.cos(angles[:,1])) 
        distances_ -= head_positions[:,2] * np.sin(angles[:,0])

        l = distances_ * np.cos(angles[:,0])
        x,y,z = (l * np.cos(angles[:,1]), l * np.sin(-angles[:,1]), distances_ * np.sin(angles[:,0]))
        
        x += head_positions[:,0]
        y += head_positions[:,1]  
      
        return np.arctan2(z, (x**2 + y**2)**0.5), np.arctan2(y, x)

    @staticmethod
    def xyz_from_heads(distances, angles, head_positions):
       
        distances_ = distances-np.cos(angles[:,0]) * (head_positions[:,1] * np.sin(angles[:,1]) + head_positions[:,0] * np.cos(angles[:,1])) 
        distances_ -= head_positions[:,2] * np.sin(angles[:,0])

        l = distances_ * np.cos(angles[:,0])
        x,y,z = (l * np.cos(angles[:,1]), l * np.sin(angles[:,1]), distances_ * np.sin(angles[:,0]))
        
        x += head_positions[:,0]
        y += head_positions[:,1]  
       
        return x,y,z

    def get_head_position_per_channel(self):
        head = np.zeros((3, self.specs['v'], self.specs['h']), dtype=np.float)
        head[:,:,:32] = self.head_positions['left'][:,None,None]
        head[:,:,32:64] = self.head_positions['center'][:,None,None]
        head[:,:,64:] = self.head_positions['right'][:,None,None]
        return np.c_[head[0].ravel(),head[1].ravel(),head[2].ravel()]