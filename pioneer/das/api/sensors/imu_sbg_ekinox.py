from pioneer.das.api.egomotion import IMUEgomotionProvider
from pioneer.das.api.interpolators import linear_ndarray_interpolator, euler_imu_linear_ndarray_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.sensors.sensor import Sensor

class ImuSbgEkinox(Sensor):

    def __init__(self, name, platform):
        factories = {key: (Sample, linear_ndarray_interpolator) for key in \
        ['gps1vel', 'timeus', 'navposvel', 'status', 'gps1hdt', 'time', 'imudata', 'gps1pos']}
        factories['ekfeuler'] = (Sample, euler_imu_linear_ndarray_interpolator)

        super(ImuSbgEkinox, self).__init__(name, platform, factories)
        
    
    def create_egomotion_provider(self):
        self.egomotion_provider = IMUEgomotionProvider(self.name, self['navposvel'], self['ekfeuler'])
        return self.egomotion_provider
