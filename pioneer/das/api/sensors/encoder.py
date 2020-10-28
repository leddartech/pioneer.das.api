from pioneer.das.api.interpolators import linear_dict_of_float_interpolator, linear_ndarray_interpolator
from pioneer.das.api.samples import RPM
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.egomotion.encoder_egomotion_provider import EncoderEgomotionProvider

class Encoder(Sensor):
    def __init__(self, name, platform):
        if not platform.is_live():
            factories = {'rpm':(RPM, linear_ndarray_interpolator)}
        else:
            factories = {'rpm':(RPM, linear_dict_of_float_interpolator)}
        super(Encoder, self).__init__(name, platform, factories)
    
    def create_egomotion_provider(self):
        self.egomotion_provider = EncoderEgomotionProvider(self.name, self)
        return self.egomotion_provider
            
