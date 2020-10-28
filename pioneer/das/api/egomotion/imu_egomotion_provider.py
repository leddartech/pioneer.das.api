from pioneer.common                                import linalg
from pioneer.das.api.datasources                   import AbstractDatasource
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider

import numpy as np

class IMUEgomotionProvider(EgomotionProvider):
    """Ego Motion provider for datasets from the RAV-4 platform"""

    def __init__(self, referential_name:str, navposvel:AbstractDatasource, ekfeuler:AbstractDatasource, subsampling:int=100):
        super(IMUEgomotionProvider, self).__init__(referential_name, subsampling)
        self.navposvel = navposvel
        self.ekfeuler = ekfeuler
        self._tf_Global_from_EgoZero = self.get_Global_from_Ego_at(self.navposvel.timestamps[0])

    def get_Global_from_Ego_at(self, ts:int, dtype = np.float64) -> np.ndarray:
        """override"""
        nav = self.navposvel.get_at_timestamp(ts).raw  # latitude, longitude, altitude
        euler = self.ekfeuler.get_at_timestamp(ts).raw # roll, pitch, yaw
        return linalg.tf_from_poseENU(linalg.imu_to_utm(nav, euler), dtype = dtype) 
