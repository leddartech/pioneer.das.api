from pioneer.das.api.interpolators   import linear_ndarray_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.sensors.sensor import Sensor
from pioneer.das.api.sensors.egomotion_provider import EgomotionProvider
from pioneer.common import linalg

import numpy as np


class CarlaEgomotionProvider(EgomotionProvider):

    def __init__(self, referential_name:str, sensor, ds_type):
        super(CarlaEgomotionProvider, self).__init__(referential_name, 1)
        self.sensor = sensor
        self.ds_type = ds_type

    def get_Global_from_Ego_at(self, ts:int, dtype=np.float64) -> np.ndarray:
        raw = self.sensor[self.ds_type].get_at_timestamp(ts).raw
        if type(raw) is dict:
            raw = raw['data']
        return linalg.tf_from_poseENU([raw['x'],raw['y'],raw['z'],raw['roll'],raw['pitch'],-raw['yaw']], dtype=dtype)

class CarlaIMU(Sensor):
    def __init__(self, name, platform):
        factories = {'agc':(Sample, linear_ndarray_interpolator)}
        super(CarlaIMU, self).__init__(name, platform, factories)

    def create_egomotion_provider(self):
        self.egomotion_provider = CarlaEgomotionProvider(self.name, self, 'agc')
        return self.egomotion_provider