from pioneer.das.api.datasources import AbstractDatasource

from typing import List, Optional, Callable, Dict, Any

import numpy as np

class VirtualDatasource(AbstractDatasource):
    def __init__(self, ds_type:str, dependencies:List[str], callback:Optional[Callable[[Dict[str, AbstractDatasource], Any], 'Sample']], timestamps_source = None):
        """Constructor
        Args:
            ds_type:        the datasource type, e.g. 'ech-bbox'
            dependencies:   a list of datasource names used to compute this datasources, e.g. ['eagle_tfc_ech', 'flir_tfc_img', ...].
            callback:       a method that will be called with arguments\:
                                'datasources'\: a dict with dependencies as keys and corresponding datasources as values
                                'key'\:  the argument passed to the virtual datasource's __getitem__().
            timestamps_source: datasource used as timestamps, If None first one dependencies used
        """
        super(VirtualDatasource, self).__init__(None, ds_type)
        self.ds_type = ds_type
        self.dependencies = dependencies
        self.datasources = None #will be configured during call to _set_sensor 
        self.callback = callback
        self.timestamps_source = timestamps_source

    def invalidate_caches(self):
        #No cache implemented for virtual datasources
        pass

    def _set_sensor(self, sensor:'Sensor'):
        """Sets this datasource's sensor"""
        self.sensor = sensor
        self.datasources = {dep:self.sensor.platform[dep] for dep in self.dependencies}

    def get_first_datasource(self) -> AbstractDatasource:
        if self.timestamps_source is None:
            for k in self.datasources:
                return self.datasources[k] #length of the first ds
        else:
            return self.datasources[self.timestamps_source]
        raise RuntimeError('no datasources')

    @property
    def timestamps(self) -> np.ndarray:
        return self.get_first_datasource().timestamps

    def get_at_timestamp(self, timestamp:int):
        """Default implementation"""

        if hasattr(self.callback, 'get_at_timestamp'):
            return self.callback.get_at_timestamp(self.datasources, timestamp)

        float_index = self.get_first_datasource().to_float_index(timestamp)
        return self[int(np.floor(float_index))]

    def __getitem__(self, key):
        return self.callback(self.datasources, key)

    @property
    def label(self):
        return "{}_{}".format(self.sensor.name, self.ds_type)
