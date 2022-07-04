from pioneer.common                                import linalg
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider
from pioneer.das.api.datasources                   import DatasourceWrapper

import numpy as np



class CarlaEgomotionProvider(EgomotionProvider):
    """Ego Motion provider for simulated datasets (CARLA simulator)"""

    def __init__(self, referential_name:str, carla_imu_datasource:DatasourceWrapper):
        super().__init__(referential_name)
        self.carla_imu_datasource = carla_imu_datasource

    def get_transform(self, timestamp:int) -> np.ndarray:
        raw = self.carla_imu_datasource.get_at_timestamp(timestamp).raw
        if type(raw) is dict: raw = raw['data']
        return linalg.tf_from_poseENU([raw['x'],raw['y'],raw['z'],raw['roll'],raw['pitch'],-raw['yaw']])

    def get_timestamps(self) -> np.ndarray:
        return self.carla_imu_datasource.timestamps
