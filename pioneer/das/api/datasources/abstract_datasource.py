from pioneer.das.api.datasources import DataSource

import numpy as np

class AbstractDatasource(DataSource):
    def __init__(self, sensor:'Sensor', ds_type:str, is_live:bool = False):
        self.sensor = sensor
        self.ds_type = ds_type
        self.config = {}
        self.is_live = is_live


    def __len__(self) -> int:
        return len(self.timestamps)

    def to_positive_index(self, index):
        if index < 0:
            index = index % len(self)
        return index
        
    @property
    def timestamps(self) -> np.ndarray:
        raise RuntimeError("Not implemented")

    @property
    def time_of_issues(self) -> np.ndarray:
        raise RuntimeError("Not implemented")

    @property
    def label(self) -> str:
        return "{}_{}".format(self.sensor.name, self.ds_type)

    def to_float_index(self, timestamp:int) -> float:
        i = np.searchsorted(self.timestamps, timestamp)
        # deal with borders
        if i == 0:
            return 0
        if i == len(self.timestamps):
            return i-1
        ts_from = self.timestamps[i - 1]
        t = float(timestamp - ts_from) / (self.timestamps[i] - ts_from)
        if t == 0:
            return i - 1
        if t == 1:
            return i

        return i - 1 + t