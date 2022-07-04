from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider
from pioneer.das.api.datasources                   import DatasourceWrapper

import numpy as np



class VayaEgomotionProvider(EgomotionProvider):
    """Ego Motion provider from Vayadrive exported datasets"""

    def __init__(self, referential_name:str, matrix_datasource:DatasourceWrapper):
        super().__init__(referential_name)
        self.matrix_datasource = matrix_datasource

    def get_transform(self, timestamp:int) -> np.ndarray:
        return self.matrix_datasource.get_at_timestamp(timestamp).raw

    def get_timestamps(self) -> np.ndarray:
        return self.matrix_datasource.timestamps
        