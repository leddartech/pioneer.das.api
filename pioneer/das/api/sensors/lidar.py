from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples.point_cloud import PointCloud
from pioneer.das.api.sensors.sensor3d import Sensor3D

from typing import Any, Dict, Tuple


class Lidar(Sensor3D):
    
    def __init__(self, name: str, platform: 'Platform', factories:Dict[str, Tuple[Any, Any]] = {}):
        super().__init__(name, platform, 
            factories = {**factories,
                'xyzit': (PointCloud, nearest_interpolator), 
            })