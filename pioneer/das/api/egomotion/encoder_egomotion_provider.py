from pioneer.common                                import linalg
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider

import numpy as np

class EncoderEgomotionProvider(EgomotionProvider):
    """Ego Motion provider for datasets from the Wheel-E platform"""

    def __init__(self, referential_name:str, sensor):
        super(EncoderEgomotionProvider, self).__init__(referential_name, 1)
        self.sensor = sensor

    def get_Global_from_Ego_at(self, ts:int, dtype=np.float64) -> np.ndarray:
        raw = self.sensor['rpm'].get_at_timestamp(ts).raw
        if type(raw) is dict:
            raw = raw['data']
        return linalg.tf_from_poseENU([raw['x'],raw['y'],raw['z'],raw['roll'],raw['pitch'],raw['yaw']], dtype=dtype)