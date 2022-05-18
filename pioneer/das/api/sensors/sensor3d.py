from pioneer.das.api.sensors.sensor  import Sensor
from pioneer.das.api.samples         import PointCloud
from pioneer.das.api.interpolators   import nearest_interpolator

from typing import Any, Dict, Tuple, Union

import numpy as np
import warnings
warnings.simplefilter('once', DeprecationWarning)


class Sensor3D(Sensor):
    
    def __init__(self, name: str, platform: 'Platform', factories:Dict[str, Tuple[Any, Any]] = {}):
        super().__init__(name, platform, factories={**factories, 'pcloud': (PointCloud, nearest_interpolator)})

    def load_intrinsics(self, intrinsics_config: Union[str, dict]):
        '''load data from yml platform '''
        if isinstance(intrinsics_config, str):
            super().load_intrinsics(intrinsics_config)
        orientation = intrinsics_config.get('orientation')
        if orientation: self.orientation = np.array(orientation)