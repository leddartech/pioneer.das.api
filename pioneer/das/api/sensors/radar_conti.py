from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples import Sample, XYZVI
from pioneer.das.api.sensors.sensor import Sensor


class RadarConti(Sensor):
    def __init__(self, name, platform):
        super(RadarConti, self).__init__(name,
                                      platform,
                                      {'xyzvi': (XYZVI, nearest_interpolator)})

        self.amplitude_type = 'velocity'  # types = ['i', 'velocity']

    def get_corrected_cloud(self, _timestamp, pts, _dtype):
        return pts
