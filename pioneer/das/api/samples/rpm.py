from pioneer.das.api.samples.sample import Sample

import numpy as np

class RPM(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(RPM, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def meters_per_second(self):
        rpm = self.raw['data']
        rps = np.array([rpm['left']/60, rpm['right']/60])*2.0
        try:
            return rps*self.datasource.sensor.yml['wheel_diameter']*np.pi
        except:
            raise ValueError("wheel_diameter has to be in the encoder's yml in order to convert rpm data")  
        
