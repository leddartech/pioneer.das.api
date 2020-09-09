from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.samples import Sample
from pioneer.das.api.interpolators   import linear_ndarray_interpolator

class CarlaGPS(Sensor):
    def __init__(self, name, platform):
        factories = {'pos':(Sample, linear_ndarray_interpolator)}
        super(CarlaGPS, self).__init__(name, platform, factories)
