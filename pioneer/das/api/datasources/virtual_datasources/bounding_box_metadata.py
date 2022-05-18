from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Sample

from typing import Any, List, Optional

import numpy as np


class BoundingBoxMetadata(VirtualDatasource):
    """Extract metadata from bounding boxes"""

    def __init__(self, reference_sensor:str, dependencies:List[str], category_counts:Optional[List[str]]=None):
        """Constructor
            Args:
                reference_sensor (str): The name of the reference sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the bounding boxes datasources. 
        """
        ds_type = 'scalars'
        super(BoundingBoxMetadata, self).__init__(ds_type, dependencies, None)
        self.reference_sensor = reference_sensor  
        self.reference_datasource = dependencies[0]
        self.category_counts = category_counts

        dtype = []
        
        if self.category_counts:
            dtype += [(category_name, 'u8') for category_name in self.category_counts]

        self.dtype = np.dtype(dtype)

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.reference_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    def __getitem__(self, key:Any):

        try: #is key iterable?
            return [self[index] for index in key]
        except: pass

        raw = np.zeros(1, self.dtype)

        for ds_name in self.dependencies:
            sample = self.datasources[ds_name][key]

            for category_name in sample.get_categories():
                if category_name in self.category_counts: raw[category_name] += 1

        return Sample(index=key, datasource=self, virtual_raw=raw, virtual_ts=sample.timestamp)

