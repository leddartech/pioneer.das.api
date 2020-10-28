from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.sensors.pixell import Pixell
from pioneer.das.api.sensors.motor_lidar import MotorLidar
from pioneer.das.api.sensors.camera import Camera
from pioneer.das.api.sensors.imu_sbg_ekinox import ImuSbgEkinox
from pioneer.das.api.sensors.encoder import Encoder
from pioneer.das.api.sensors.carla_imu import CarlaIMU 
from pioneer.das.api.sensors.radar_ti import RadarTI
from pioneer.das.api.sensors.lcax import LCAx
from pioneer.das.api.sensors.lca3 import LCA3


SENSOR_FACTORY = {  
    'lca2': LCAx,
    'pixell': Pixell,
    'lca3': LCA3,
    'eagle': LCA3,
    'flir': Camera,
    'camera': Camera,
    'sbgekinox': ImuSbgEkinox,
    'vlp16': MotorLidar,
    'ouster64': MotorLidar,
    'peakcan': Sensor,
    'radarTI': RadarTI,
    'webcam': Camera,
    'encoder': Encoder,
    'carlaimu': CarlaIMU,
    'leddar': LCAx,
    'lidar': MotorLidar,
    'any': Sensor,
}