from pioneer.das.api.interpolators   import linear_ndarray_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.egomotion.carla_egomotion_provider import CarlaEgomotionProvider

class CarlaIMU(Sensor):
    def __init__(self, name, platform):
        factories = {'agc':(Sample, linear_ndarray_interpolator)}
        super(CarlaIMU, self).__init__(name, platform, factories)

    def create_egomotion_provider(self):
        self.egomotion_provider = CarlaEgomotionProvider(self.name, self)
        return self.egomotion_provider