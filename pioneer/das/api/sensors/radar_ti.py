from pioneer.das.api.interpolators   import nearest_interpolator
from pioneer.das.api.samples         import Sample, XYVIT
from pioneer.das.api.sensors.sensor  import Sensor

class RadarTI(Sensor):
    def __init__(self, name, platform):
        super(RadarTI, self).__init__(name
                                , platform
                                , {  'rad': (XYVIT, nearest_interpolator)
                                    ,'rec': (Sample, nearest_interpolator)
                                    ,'rtr': (Sample, nearest_interpolator)})

    def get_corrected_cloud(self, timestamp, pts, dtype):
        return pts
