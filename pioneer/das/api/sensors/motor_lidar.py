from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.sensors.lidar  import Lidar
from pioneer.das.api.samples        import Sample, XYZIT
from pioneer.das.api.interpolators  import nearest_interpolator

from typing import Any, Dict, Tuple, Union

import numpy as np

class MotorLidar(Lidar):
    
    def __init__(self, name: str, platform: 'Platform', factories:Dict[str, Tuple[Any, Any]] = {}):
        super().__init__(name, platform, 
            factories = {**factories,
                'xyzit': (XYZIT, nearest_interpolator), 
                'temp': (Sample, nearest_interpolator),
            })
        
        self.temperature_slope = None
        self.temperature_reference = None
        self.temperature_coeffs = None
        self.temperature_compensation = False
        
        self.pcl_datasource = 'xyzit'

    def load_intrinsics(self, intrinsics_config: Union[str, dict]):
        '''load data from yml platform '''
        temperature_config = intrinsics_config.get('temperature')
        if temperature_config:
            self.temperature_compensation = True
            self.temperature_slope = temperature_config['slope']
            self.temperature_reference = temperature_config['reference']
            self.temperature_coeffs = np.array([self.temperature_slope, 
                                            self.temperature_reference])
        
        orientation = intrinsics_config.get('orientation')
        if orientation:
            self.orientation = np.array(orientation)

    def apply_temperature_correction(self, timestamp, distances):
        """Applies temperature-related distance corrections"""
        if (self.temperature_slope is None and
            self.temperature_reference is None):
            self.temperature_compensation = False
            LoggingManager.instance().warning('MotorLidar Log: Temperature compensation mode is deactivated')
            return distances
        
        offsets_ = 0
        try:
            temperature = self.datasources['temp'].get_at_timestamp(timestamp).raw['data']
            k = len(self.temperature_coeffs)-1
            offsets_ -= np.sum([temperature**(k-i) * self.temperature_coeffs[i] for i in range(k)])
        except Exception as e:
            temperature = None
            LoggingManager.instance().warning('Failed to apply the temperature for the distance correction. '
                                'Error: {}'.format(str(e)))
        offsets_ -= self.temperature_coeffs[-1]

        #print(f'Log: 3D points for {self.name} are corrected with temperature T={temperature}: global offset is {offsets_}')
        return distances + offsets_
        
    def get_corrected_cloud(self, timestamp, pts, dtype):
        '''It corrects the pts-cloud according to a temperature compensation (if any).'''
        if not self.temperature_compensation:
            return pts
        distances = np.linalg.norm(pts, axis=1, keepdims=True)
        directions = pts / distances
        distances = self.apply_temperature_correction(timestamp, distances)
        pts = directions * distances
        return pts
