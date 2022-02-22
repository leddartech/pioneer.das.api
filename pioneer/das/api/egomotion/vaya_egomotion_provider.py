from pioneer.common                                import linalg
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider

import numpy as np

class VayaEgomotionProvider(EgomotionProvider):
    """Ego Motion provider from Vayadrive exported datasets"""

    def __init__(self, referential_name:str, sensor):
        super(VayaEgomotionProvider, self).__init__(referential_name, 1)
        self.sensor = sensor

    def get_Global_from_Ego_at(self, ts:int, dtype=np.float64) -> np.ndarray:
        return self.sensor['mat'].get_at_timestamp(ts).raw