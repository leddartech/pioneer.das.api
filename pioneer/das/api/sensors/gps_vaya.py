from pioneer.das.api.interpolators   import linear_ndarray_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.egomotion.vaya_egomotion_provider import VayaEgomotionProvider

# From the vayavision egomotion export as raw 4x4 matrix
class GPSVaya(Sensor):
    def __init__(self, name: str, platform: 'Platform'):
        factories = {'mat':(Sample, linear_ndarray_interpolator)}
        super(GPSVaya, self).__init__(name, platform, factories)

    def create_egomotion_provider(self):
        self.egomotion_provider = VayaEgomotionProvider(self.name, self)
        return self.egomotion_provider