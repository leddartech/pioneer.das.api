from pioneer.das.api.samples.point_cloud import PointCloud

from typing import Optional, Tuple

import numpy as np


class XYZV(PointCloud):
    """Point cloud provided by a radar"""

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super().__init__(index, datasource, virtual_raw, virtual_ts)

    @property
    def fields(self) -> Tuple[str]:
        return self.raw[0].dtype.names

    def get_field(self, field: str) -> Optional[np.ndarray]:
        if field in self.fields: 
            return self.raw[0][field]

    @property
    def size(self) -> int:
        return self.raw[0].shape[0]