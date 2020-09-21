from pioneer.common import banks, clouds, constants
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.sensors.lcax import LCAx

from enum import Enum
from typing import Callable, Union, Optional, List, Dict, Tuple, Any

import numpy as np




def from_spherical_to_cartesian(distance, elevation, azimut):
    l = distance * np.cos(elevation)
    return (
        l * np.cos(azimut), 
        l * np.sin(azimut),
        distance * np.sin(elevation) 
    )

def referential_transform(distance, elevation, azimut, a, b, c):
    '''Return mappeds (distance, elevation, azimut) in a new referential.
        The parameters (a,b,c) are some constants.
    '''
    x_, y_, z_ = from_spherical_to_cartesian(distance, elevation, azimut)
    x_ += a
    y_ += b
    z_ += c
    return (
        (x_**2 + y_**2 + z_**2)**0.5,
        np.arctan2(z_, (x_**2 + y_**2)**0.5),
        np.arctan2(y_, x_)
    )

def from_calibration_jig_to_sensor(pitch, yaw, pivot):
    '''Input data from the calibration jig are converted to sensor data at 
        universal point referential (elevation, azimut)
    '''
    return (
        np.arctan2(np.sin(pitch) * np.cos(yaw - pivot) , (np.sin(yaw - pivot)**2 + np.cos(pitch)**2 * np.cos(yaw - pivot)**2)**0.5),
        np.arctan2(np.tan(yaw) , np.cos(pitch))
    )

def from_sensor_to_lcas(distance, elevation, azimut, dz_elevation, dx_azimut, dy_azimut):
    '''Convert a coordinate (distance, elevation, azimut) from universal
        sensor point to lcas sub-module.
        
        Constants:
            dz_elevation: is the distance on the z-axis (or along elevation) between sensor point and lcas point
            dx_azimut: distance on the x-axis between referentials
            dy_azimut: distance on the y-axis between referentials
    '''
    return referential_transform(distance, elevation, azimut, -dx_azimut, -dy_azimut, -dz_elevation)

def from_lcas_to_sensor(distance, elevation, azimut, dz_elevation, dx_azimut, dy_azimut):
    '''Convert a coordinate (distance, elevation, azimut) from submodule
        lcas point to unviversal sensor point.

        Constans: (see: from_sensor_to_lcas)
    '''
    return referential_transform(distance, elevation, azimut, dx_azimut, dy_azimut, dz_elevation)

def sensor_angles_from_lcas_angles(distance, elevation, azimut, dx_azimut, dy_azimut):
    '''Compute (elevation, azimut) in sensor referential from
            distance: from sensor referential
            elevation: from lcas referential
            azimut: from lcas referential
        and
        Constants: (see from_sensor_to_lcas)
    '''
    gamma_ = np.cos(elevation) * (dy_azimut * np.sin(azimut) + dx_azimut * np.cos(azimut))
    distance_ = - gamma_ + (gamma_**2  + distance**2 - dx_azimut**2 - dy_azimut**2)**0.5
    _, elevation_, azimut_ = from_lcas_to_sensor(distance_, elevation, azimut, 0, dx_azimut, dy_azimut)
    return elevation_, azimut_




class Pixell(LCAx):
    class IntrinsicMode(Enum):
        Spherical = 0
        Calibration = 1
    
    class CloudMode(Enum):
        Point = 'point_cloud'
        Quad = 'quad_cloud'

    class SubLcaModules(Enum):
        LCA1 = 0
        LCA2 = 1
        LCA3 = 2
   
    class SensorAngles(Enum):
        Elevation = 0
        Azimut = 1

    class CalibJigAngles(Enum):
        Pitch = 0
        Yaw = 1
        Pivot = 2
    
    class FastTraceType(Enum):
        LowRange = 256
        MidRange = 512

    # TODO: move all this in constants.py
    LCAS_OFFSETS = {SubLcaModules.LCA1: np.array([0.03479,  0.05655, 0.01562]),
                    SubLcaModules.LCA2: np.array([0.03952,  0,       0.01562]),
                    SubLcaModules.LCA3: np.array([0.03479, -0.05655, 0.01562])
                    }
    CALIB_TARGET_DISTANCE = {SensorAngles.Elevation: 3.5,
                            SensorAngles.Azimut: 6.88
                            }
    CALIB_JIG_PIVOT = np.deg2rad(np.hstack((np.full(11, 80.0),
                                            np.full(10, 60.0),
                                            np.full(11, 40.0),
                                            np.full(11, 20.0),
                                            np.full(10, 0.0),
                                            np.full(11, -20.0),
                                            np.full(11, -40.0),
                                            np.full(10, -60.0),
                                            np.full(11, -80.0))).astype(float)
                                )

    def __init__(self, name, platform):
        super(Pixell, self).__init__(name, platform)
        self.orientation = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0,-1]], 'f8')
        self.intrinsic_mode = Pixell.IntrinsicMode.Calibration #Default mode
        self.has_lcas_angles_calibration = True #The new sdk has the module angles hardcoded, hence falling back on this approach.
        self.modules_angles = None
        self.adc_freq = 100e6
    
    def __set_spherical_mode(self):
        '''Set the spherical mode of Pixell when no calibration data is provided. 
        '''
        self.intrinsic_mode = Pixell.IntrinsicMode.Spherical
        if self.projection is None:
            self.projection = 'direction_spherical'
        
        self.orientation = np.array([[0 , 0, 1],
                                     [-1, 0, 0],
                                     [0 ,-1, 0]], 'f8')
        self.cache(self.specs)

    def load_intrinsics(self, intrinsics_config: Union[str, dict]):
        '''Basic intrinsics are brougth back under Sensor.
        '''
        super(Pixell, self).load_intrinsics(intrinsics_config)
        try:
            cfg = self['cfg'][0].raw
            try:
                if self.specs is None:
                    self.specs = banks.extract_specs(lambda n: cfg[n])
            
                if self.modules_angles is None:
                    try:
                        self.modules_angles = banks.extract_intrinsics_modules_angles(lambda n: cfg[n])
                    except:
                        LoggingManager.instance().warning("Module angles calibration data not accessible")
                        self.__set_spherical_mode()

                if self.time_base_delays is None:
                    self.time_base_delays = banks.extract_intrinsics_timebase_delays(lambda n: cfg[n])
            except:
                LoggingManager.instance().warning("Unable to read completely data from cfg file")
        except:
            LoggingManager.instance().warning("No configuration file cfg and calibration data is available for {}.".format(self.name))
            self.__set_spherical_mode() #safe-mode: spherical
    
    def get_modules_angles(self):
        return self._modules_angles

    def set_modules_angles(self, package):
        if package is None:
            self._modules_angles = None
        else:
            self._modules_angles = {Pixell.CalibJigAngles.Pitch: package['ID_CHANNEL_ANGLE_ELEVATION'],
                                    Pixell.CalibJigAngles.Yaw: package['ID_CHANNEL_ANGLE_AZIMUT'],
                                    Pixell.CalibJigAngles.Pivot: np.hstack([Pixell.CALIB_JIG_PIVOT for i in range(8)])
                                    }
            self.cache(self.specs) #initialize cache
            
    modules_angles = property(get_modules_angles, set_modules_angles)

    def set_time_base_delays(self, package):
        if package is None:
            self._time_base_delays = None
        else:
            if self.modules_angles is not None:
                cache_ = self.cache(self.specs)
                m_ = int(self.specs['v'] * self.specs['h'])
                self._time_base_delays = {
                    Pixell.FastTraceType.LowRange: package[:m_][constants.ID_FROM_LCAS_CHAN_TO_SENSOR_CHAN],
                    Pixell.FastTraceType.MidRange: package[m_:][constants.ID_FROM_LCAS_CHAN_TO_SENSOR_CHAN]
                }

                elevation, azimut = self.get_sensor_projection_data(Pixell.CALIB_TARGET_DISTANCE[Pixell.SensorAngles.Azimut], 
                                                                        cache_['angles'], cache_['lcas_offsets'])
                d_lcas_true, _, _ = from_sensor_to_lcas(Pixell.CALIB_TARGET_DISTANCE[Pixell.SensorAngles.Azimut], 
                                                                elevation, azimut, 0, cache_['lcas_offsets'][:,0], cache_['lcas_offsets'][:,1]
                                                                )
                for mu in Pixell.FastTraceType:
                    self._time_base_delays[mu] =  d_lcas_true - (Pixell.CALIB_TARGET_DISTANCE[Pixell.SensorAngles.Azimut] - self._time_base_delays[mu])

                print("Pixell Log: time base delays are converted to lcas referential.")

    def get_time_base_delays(self):
        return self._time_base_delays
        
    time_base_delays = property(get_time_base_delays, set_time_base_delays)

    def cache(self, specs):
        """ Derivation of cache for Pixell sensor  
        """
        hashable = frozenset([(k,specs[k]) for k in ['v', 'h', 'v_fov', 'h_fov']])
        if not hashable in self._cache:
            cache = super(Pixell, self).cache(specs)
            cache['lcas_offsets'] = self.__get_lcas_offsets_per_channel()
            if self.modules_angles is not None:
                cache['angles'] = self.__get_lcas_directions(cache['lcas_offsets'])
                cache['quad_angles'] = clouds.custom_quad_angles(specs, cache['angles'])
                print('Pixell Log: the cache is set to use calibration angles.')
            else:
                cache['quad_angles'] = clouds.quad_angles(specs)
                print(f'Pixell Log: the cache is set with uncalibrated directions ({self.projection})')
            self._cache[hashable] = cache

        return self._cache[hashable]

    def __get_lcas_offsets_per_channel(self):
        _offsets = np.zeros((3,8,96), dtype=np.float)
        for i in range(3):
            for _lca in Pixell.SubLcaModules:
                _offsets[i, :, int(_lca.value*32):int((_lca.value+1)*32)] = Pixell.LCAS_OFFSETS[_lca][i]
        return np.c_[
            _offsets[0].ravel(),
            _offsets[1].ravel(),
            _offsets[2].ravel()
        ]
    
    def __get_lcas_directions(self, offsets):
        '''Compute the (elevation, azimut) for each channel, based on intrinsic calibration data.
        '''
        if self.has_lcas_angles_calibration:
            elevation = self.modules_angles[Pixell.CalibJigAngles.Pitch]
            azimut = self.modules_angles[Pixell.CalibJigAngles.Yaw]
            return np.c_[elevation, azimut].astype(np.float)

        phi, theta = from_calibration_jig_to_sensor(self.modules_angles[Pixell.CalibJigAngles.Pitch],
                                                        self.modules_angles[Pixell.CalibJigAngles.Yaw],
                                                        self.modules_angles[Pixell.CalibJigAngles.Pivot])
        _, elevation, _ = from_sensor_to_lcas(Pixell.CALIB_TARGET_DISTANCE[Pixell.SensorAngles.Elevation], 
                                                  phi, theta, offsets[:,2], offsets[:,0], offsets[:,1])
        _, _, azimut = from_sensor_to_lcas(Pixell.CALIB_TARGET_DISTANCE[Pixell.SensorAngles.Azimut], 
                                               phi, theta, offsets[:,2], offsets[:,0], offsets[:,1])
        return np.c_[elevation, azimut].astype(np.float)

    def get_cloud(self, cache, type, indices, distances, amplitudes=None, dtype=np.float64):
        '''Get cloud data (type_pts, type_amplitudes, type_indices) from a sensor.
        '''
        if type is Pixell.CloudMode.Point.value:
            elevation, azimut = self.get_sensor_projection_data(distances, cache['angles'][indices], cache['lcas_offsets'][indices])
            y,z,x = clouds.direction_spherical(elevation, azimut) #this system is broken...
            pts = np.stack((distances * x, distances * y, distances * z), axis=-1)
            return pts, amplitudes, indices
        
        elif type is Pixell.CloudMode.Quad.value:
            distances_, elevation, azimut = [],[],[]
            m_ = cache['corrected_specs']['v'] * cache['corrected_specs']['h']
            for i in range(4):
                elevation_, azimut_ = self.get_sensor_projection_data(distances, cache['quad_angles'][indices + i * m_], 
                                                                        cache['lcas_offsets'][indices])
                distances_.extend(distances)
                elevation.extend(elevation_)
                azimut.extend(azimut_)
            distances_ = np.array(distances_, dtype=np.float)
            elevation = np.array(elevation, dtype=np.float)
            azimut = np.array(azimut, dtype=np.float)
            y,z,x = clouds.direction_spherical(elevation, azimut) #this system is broken...
            pts = np.stack((distances_ * x, distances_ * y, distances_ * z), axis=-1)
            quad_amplitudes = clouds.quad_stack(amplitudes)
            quad_indices = clouds.generate_quads_indices(indices.shape[0], np.uint32).flatten()
            return pts, quad_amplitudes, quad_indices
    
    def get_sensor_projection_data(self, distances, angles, offsets):
        '''Compute (elevation, azimut) in the sensor referential from
            the distance at sensor referential
        '''
        return sensor_angles_from_lcas_angles(distances, angles[:,0], angles[:,1], offsets[:,0], offsets[:,1])

    def get_corrected_cloud(self, timestamp, cache, type, indices, distances, amplitudes=None, dtype=np.float64):
        
        if self.intrinsic_mode == Pixell.IntrinsicMode.Spherical:
            return super(Pixell, self).get_corrected_cloud(timestamp, cache, type, indices, distances, amplitudes, dtype)

        elif self.intrinsic_mode == Pixell.IntrinsicMode.Calibration:
            type_pts, type_amplitudes, type_indices = self.get_cloud(cache, type, indices, distances, amplitudes, dtype)

            if type is Pixell.CloudMode.Point.value:
                return type_pts
            elif type is Pixell.CloudMode.Quad.value:
                return type_pts, type_amplitudes, type_indices
            else:
                raise KeyError('Can not compute the point cloud projection')
    
    def get_fast_trace_smoothing_kernel(self, fast_trace_type, scaling=1):
        '''Returns the convolution kernel filter
            fast_trace_type is one of FastTraceType
        '''
        return constants.FILTERS_KERNEL[fast_trace_type.value][self.oversampling]*scaling
