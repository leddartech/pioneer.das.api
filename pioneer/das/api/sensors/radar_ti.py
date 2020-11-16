from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples import Sample, XYZVCFAR
from pioneer.das.api.sensors.sensor import Sensor


class RadarTI(Sensor):
    def __init__(self, name, platform):
        super(RadarTI, self).__init__(name,
                                      platform,
                                      {'rec': (Sample, nearest_interpolator),
                                       'rtr': (Sample, nearest_interpolator),
                                       'xyzvcfar': (XYZVCFAR, nearest_interpolator)})

        # XYZVCFAR sample use this parameter to determine the amplitude type to return in 'amplitudes'.
        # Therefore, this sensor is not stateless
        self.amplitude_type = 'cfar_snr'  # types = ['cfar_snr', 'cfar_noise', 'velocity']

    def get_corrected_cloud(self, _timestamp, pts, _dtype):
        return pts
