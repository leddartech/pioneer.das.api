from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.sensors.sensor3d import Sensor3D
from pioneer.das.api.sensors.lidar import Lidar
from pioneer.das.api.sensors.radar import Radar
from pioneer.das.api.sensors.pixell import Pixell
from pioneer.das.api.sensors.motor_lidar import MotorLidar
from pioneer.das.api.sensors.camera import Camera
from pioneer.das.api.sensors.imu_sbg_ekinox import ImuSbgEkinox
from pioneer.das.api.sensors.carla_imu import CarlaIMU 
from pioneer.das.api.sensors.lcax import LCAx
from pioneer.das.api.sensors.lca3 import LCA3
from pioneer.das.api.sensors.gps_vaya import GPSVaya



SENSOR_FACTORY = {  
    'lca2': LCAx,
    'pixell': Pixell,
    'lca3': LCA3,
    'eagle': LCA3,
    'flir': Camera,
    'camera': Camera,
    'sbgekinox': ImuSbgEkinox,
    'vlp16': Lidar,
    'ouster64': MotorLidar,
    'radarTI': Radar,
    'radarConti': Radar,
    'webcam': Camera,
    'carlaimu': CarlaIMU,
    'leddar': LCAx,
    'lidar': Lidar,
    'gpsvaya': GPSVaya,
}