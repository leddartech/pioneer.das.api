from pioneer.common import clouds, banks, misc
from pioneer.common.logging_manager import LoggingManager
from pioneer.common.types import calibration
from pioneer.das.api.interpolators import nearest_interpolator, linear_dict_of_float_interpolator, floor_interpolator
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.samples import Sample, Echo, Trace, FastTrace, EchoXYZIT

import numpy as np
from typing import Callable, Union, Optional, List, Dict, Tuple, Any

class LCAx(Sensor):
    """ LeddarTech LCAx family of sensors, expects 'ech', 'sta' and 'cfg' datasources"""

    def __init__(self, name, platform):
        super(LCAx, self).__init__(name
                                   , platform
                                   , {  'ech': (Echo, nearest_interpolator)
                                      , 'sta': (Sample, linear_dict_of_float_interpolator)
                                      , 'cfg': (Sample, floor_interpolator)
                                      , 'trr': (Trace, nearest_interpolator)
                                      , 'ftrr': (FastTrace, nearest_interpolator)
                                      , 'xyzit': (EchoXYZIT, nearest_interpolator)
                                     }
                                   )

        self._cache = {}
        # intrinsics override
        self.specs = None
        self.temperature_slope = None
        self.temperature_reference = None
        self.time_base_delays = None
        self.static_noise = None
        self.angle_chart = None
        self.angle_chart_path = None
        self.angle_chart_ref_v_fov = None
        self.mirror_temp_compensation = None
        self.factor_correction = None
        self.projection = None
        self.config = {'reject_flags': [3, 3072]
        , 'extrema_amp' : 2**20 #2d images only
        , 'extrema_dist': 250 #2d images only
        , 'dist_reject_intervals': []
        , 'amp_reject_intervals': []}
        self.scan_direction = 'vertical'
        self.saturation_calibration = None
        self.orientation = np.array([[ 1,  0,  0],
                                     [ 0,  1,  0],
                                     [ 0,  0,  1]], 'f8')
        self.pcl_datasource = 'ech'


    def cache(self, specs: Dict[str, Union[int, float]]) -> Dict[str, np.ndarray]:
        """ caches computation that are shared for all 'ech' samples.
        
        Args:
            specs: a dict with keys, 'v'\: number of scan channels, 'h'\: the number of imaging channels,
                'v_fov'\: scan field of view, in degrees, 'h_fov'\: imaging field of view, in degrees

        Returns: 
            A dict with keys 'angles' (the projection angles), 'directions' (the projection directions) 
            and 'quad_directions' (the projection surface cloud directions)
        Note:
            If 'angle_chart' is defined in this sensor's intrinsics, the angle chart's angles is used to
            compute 'directions', but **not** 'quad_directions'.
            If 'mirror_temp_compensation' is defined in this sensor's intrisincs, 'angles_temp_lut' 
            and 'directions_temp_lut' will appear in cache and be used by apply_direction_corrections()

        """

        hashable = frozenset([(k,specs[k]) for k in ['v', 'h', 'v_fov', 'h_fov']]) #order invariant set

        if not hashable in self._cache:
            cache = {}

            if self.angle_chart is not None :
                specs = dict(specs)
                ## To be coherent with angle in chart table
                if self.angle_chart_ref_v_fov is not None : 
                    specs['v_fov'] = self.angle_chart_ref_v_fov

                if self.mirror_temp_compensation is not None:
                    angles_temp_lut = {}
                    directions_temp_lut = {}
                    quad_directions_temp_lut = {}
                    
                    temp_min, m, b = self.mirror_temp_compensation['t_min'], self.mirror_temp_compensation['slope'], self.mirror_temp_compensation['y0']

                    for t in np.arange(round(temp_min), 60, 0.5):
                        f = m*t + b
                        a = angles_temp_lut[t] = clouds.custom_v_angles(specs, factor = f, filename = self.angle_chart_path)
                        directions_temp_lut[t] = clouds.directions(a)
                        quad_directions_temp_lut[t] = clouds.custom_v_quad_directions(specs, factor = f, filename = self.angle_chart_path)

                    cache['angles_temp_lut'] = angles_temp_lut
                    cache['directions_temp_lut'] = directions_temp_lut
                    cache['quad_directions_temp_lut'] = quad_directions_temp_lut
                    
                cache['angles'] = clouds.custom_v_angles(specs, factor = self.factor_correction, filename = self.angle_chart_path)
                cache['directions'] = clouds.directions(cache['angles'])
                cache['quad_directions'] = clouds.custom_v_quad_directions(specs, factor = self.factor_correction, filename = self.angle_chart_path)

            else:
                if self.projection == 'direction_orthogonal':
                    cache['angles'] = clouds.angles(specs)
                    cache['directions'] = clouds.directions(cache['angles'],direction_f=getattr(clouds,self.projection))
                    cache['quad_directions'] = clouds.quad_directions(specs,direction_f=getattr(clouds,self.projection))
                elif self.projection == 'direction_carla':
                    cache['angles'] = clouds.angles(specs)
                    cache['directions'],cache['quad_directions'] = clouds.directions_orthogonal(specs)
                elif self.projection == 'direction_carla_pixell':
                    cache['angles'] = clouds.angles(specs)
                    cache['directions'],cache['quad_directions'] = clouds.directions_orthogonal_pixell(specs)
                elif self.projection == 'direction_spherical':
                    cache['angles'] = clouds.angles(specs)
                    cache['directions'] = clouds.directions(cache['angles'], direction_f=clouds.direction_spherical)
                    cache['quad_directions'] = clouds.quad_directions(specs, direction_f=clouds.direction_spherical)                    
                else: #default mode
                    cache['angles'] = clouds.angles(specs)
                    cache['directions'] = clouds.directions(cache['angles'], direction_f=clouds.direction)
                    cache['quad_directions'] = clouds.quad_directions(specs, direction_f=clouds.direction)
            
            cache['corrected_specs'] = specs
            
            self._cache[hashable] = cache
       
        return self._cache[hashable]

    def load_intrinsics(self, intrinsics_config: Union[str, dict]):
        """Load the LCAx intrinsics from a dictionnary like {'v': 8, 'h', 32, 'v_fov': 20, 'h_fov': 30}

        Args:
            intrinsics_config: The intrinsics configuration
        """
        if isinstance(intrinsics_config, dict):
            intrinsics_folder = intrinsics_config.get('folder', None)
            if intrinsics_folder is not None:
                super(LCAx, self).load_intrinsics(intrinsics_folder)

            specs_config = intrinsics_config.get('specs', None)
            if specs_config is not None:
                self.specs = banks.fill_specs(**intrinsics_config.get('specs', None))
            else:
                self.load_specs_from_cfg()
            self.angle_chart = intrinsics_config.get('angle_chart', None)
            if (self.angle_chart is not None) and (not isinstance(self.angle_chart,str)): ## Old platform still have path written in string directly
                self.angle_chart_path = self.angle_chart['path']
                self.angle_chart_ref_v_fov = self.angle_chart['ref_v_fov']
            else :
                self.angle_chart_path = self.angle_chart
            self.factor_correction = intrinsics_config.get('factor_correction', 1)
            self.mirror_temp_compensation = intrinsics_config.get('mirror_temp_compensation', None)
            self.projection = intrinsics_config.get('projection', None)
            self.adc_freq = float(intrinsics_config.get('adc_frequency', 100e6))
            temp_config = intrinsics_config.get('temperature', None)
            if temp_config:
                self.temperature_slope = temp_config['slope']
                self.temperature_reference = temp_config['reference']

            # self.load_static_noise_from_cfg()
            self.load_time_base_delays_from_cfg()

            saturation_calibration_config = intrinsics_config.get('saturation_calibration', None)
            if saturation_calibration_config is not None:
                self.saturation_calibration = calibration.SaturationCalibration(
                                                self.yml['intrinsics']['saturation_calibration']['distance_coefficient'],
                                                self.yml['intrinsics']['saturation_calibration']['amplitude_coefficients'])

        else:
            super(LCAx, self).load_intrinsics(intrinsics_config)

    def load_specs_from_cfg(self):
        try:
            cfg = self['cfg'][0].raw
            self.specs = banks.extract_specs(lambda n: cfg[n])
        except:
            LoggingManager.instance().warning("Unable to read the specs from the sensor {}".format(self.name))

    def load_time_base_delays_from_cfg(self):
        try:
            cfg = self['cfg'][0].raw
            self.time_base_delays = banks.extract_intrinsics_timebase_delays(lambda n: cfg[n])
        except:
            self.time_base_delays = 0 if not 'ftrr' in self else {'high':0, 'low':0}
            LoggingManager.instance().warning("Unable to read the time base delays from the sensor {}".format(self.name))
                
    def load_static_noise_from_cfg(self):
        try:
            cfg = self['cfg'][0].raw
            self.static_noise = banks.extract_intrinsics_static_noise(lambda n: cfg[n])
        except:
            self.static_noise = 0
            LoggingManager.instance().warning("Unable to read the static noise from the sensor {}".format(self.name))

    def get_corrected_projection_data(self, timestamp:Union[int, float], cache:dict, type:str='directions'):
        """ Returns temperature compensated projections directions or angles

        Args:
            timestamp:  the timestamp at which the sample is needed (use to obtain the right temperature)
            cache:      the cache to search in (obtained by calling LCAx.cache(...))
            type:       the type of directions (e.g) 'directions' or 'quad_directions'. 
                        **Must be present in the cache**

        Returns:
            the compensated directions if any, or the uncompensated direction otherwise
        """

        default_directions = cache[type]

        if self.mirror_temp_compensation is not None:
            try:
                temp = self.get_temperature_at(timestamp)
            except Exception as e:
                LoggingManager.instance().warning(f"Failed to apply temperature correction. Error: {str(e)}")
                return default_directions
            
            temp = min(max(temp, self.mirror_temp_compensation['t_min']), 60)
            temp_floor = np.floor(temp)
            if (temp - temp_floor) >= 0.5:
                temp = temp_floor + 0.5
            else:
                temp = temp_floor

            return cache[f'{type}_temp_lut'][temp]
            
        return default_directions
    
    def get_corrected_cloud(self, timestamp, cache, type, indices, distances, amplitudes=None, dtype=np.float64):
        """Returns the point-cloud (or quad-cloud) using the get_corrected_projection_data() method.

            Args:
                type: 'point_cloud' or 'quad_cloud'

        """
        if type == 'point_cloud':
            directions_ = self.get_corrected_projection_data(timestamp, cache, type='directions')
            return clouds.to_point_cloud(indices, distances, directions_, dtype)
        elif type == 'quad_cloud':
            quad_directions_ = self.get_corrected_projection_data(timestamp, cache, type='quad_directions')
            return clouds.to_quad_cloud(indices, distances, amplitudes, quad_directions_, cache['corrected_specs']['v'],cache['corrected_specs']['h'], dtype)
        else:
            raise KeyError('Can not compute the point cloud projection')
        
    def apply_distance_corrections(self, timestamp, indices, distances):
        """Applies calibration and temperature-related distances corrections"""
        distances = self.apply_temperature_correction(timestamp, indices, distances)
        return distances

    @property
    def oversampling(self):
        try: return int(self.yml['configurations']['frequency']['ID_OVERSAMPLING_EXP'])
        except: return 1

    @property
    def binning(self):
        try: return int(self.yml['configurations']['frequency']['binning'])
        except: return 1

    @property
    def base_point_count(self):
        try: return int(self.yml['configurations']['frequency']['ID_BASE_POINT_COUNT'])
        except: bpc = int(self.platform[f'{self.name}_cfg'][0].raw['ID_BASE_POINT_COUNT'])
        return int(bpc/self.binning)

    @property
    def distance_scaling(self):
        """Get the distance in meters between two subsequent points in a trace"""
        return misc.distance_per_sample(self.adc_freq, self.oversampling, self.binning)

    @property
    def trace_smoothing_kernel(self):
        return self.get_trace_smoothing_kernel()

    def get_trace_smoothing_kernel(self):
        """Get the convolution kernel for trace processing for LCA2"""
        kernel = [71,114,71]
        if self.oversampling == 2:
            kernel = [18,34,48,56,48,34,18]
        elif self.oversampling == 4:
            kernel = [6,9,13,17,20,24,26,27,26,24,20,17,13,9,5]
        elif self.oversampling == 8:
            kernel = [4,6,8,11,13,16,18,20,21,22,21,20,18,16,13,11,8,6,4]
        return np.array(kernel)/np.sum(kernel)

    def get_temperature_at(self, timestamp):
        sample = self.datasources['sta'].get_at_timestamp(timestamp, interpolator=floor_interpolator)
        state = sample.raw
        return state['apd_temp'][0]
                 
    def apply_temperature_correction(self, timestamp, indices, distances):
        """Applies temperature-related distance corrections"""
        if (self.temperature_slope is not None and
            self.temperature_reference is not None):
            try:
                temp = self.get_temperature_at(timestamp)
            except Exception as e:
                LoggingManager.instance().warning('Failed to apply temperature correction. '
                                'Error: {}'.format(str(e)))
                return distances

            temp_offset = self.temperature_slope * (temp - self.temperature_reference)
            return distances + temp_offset

        return distances
