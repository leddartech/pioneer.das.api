from pioneer.common import misc, linalg
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.datasources import DatasourceWrapper, VirtualDatasource
from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.samples import annotations
from pioneer.das.api.sources import FileSource

from typing import Callable, Union, Optional, List, Dict, Tuple, Any

import glob
import numpy as np
import os
import pickle

class Sensor(object):
    """A sensor encapsulate a sensor and its datasources. 
    *Important* when you add a new derivation of class Sensor, 
    don't forget to add it to platform.SENSOR_FACTORY.

    """
    class NoPathToReferential(Exception):
        pass

    def __init__(self, name:str
    , pf
    , factories:Dict[str, Tuple[Any, Any]] = {}
    , call_start:Optional[Callable] = None
    , call_stop:Optional[Callable] = None):
        """Constructor.
        
        Args:
            name: the sensor's name and position id, e.g. 'eagle_tfc'
            platform: the platform this sensor belongs to
            factories: a dict containing one entry per datasource type, each entry's value is a tuple containing 
                \(Sample-derived, optional_interpolator_function\). For example, a factoryies dict could be 
                {"ech"\:(Echo, None), "sta":(Sample, interpolators.linear_dict_of_float_interpolator)}
            call_start: a callback to be called by Sensor.start() (e.g. to start an actual live sensor)
            call_stop: similar to call_start
        """
        self.name = name
        self.datasources = {}
        self.factories = {**factories, **annotations.ANNOTATIONS_FACTORY}
        self.pf = pf
        self.yml = {}
        try:
            self.yml = pf.yml[name]
        except:
            pass
        self.orientation = None
        self.extrinsics = {}
        self.intrinsics = None
        self.call_start = call_start
        self.call_stop = call_stop
        self.egomotion_provider = None
        self._extrinsics_dirty = misc.Signal()
        self.pcl_datasource = None

    def datasource_names(self) -> List[str]:
        """Returns this sensor's datasource types"""
        return [ds.label for ds in self.datasources.values()]

    def start(self):
        """Starts the live sensor wrapped by this Sensor instance """
        if self.call_start is None:
            raise RuntimeError(f"{self.name} is not a live sensor")
        self.call_start()

    def stop(self):
        """Stops the live sensor wrapped by this Sensor instance """
        if self.call_stop is None:
            raise RuntimeError(f"{self.name} is not a live sensor")
        self.call_stop()

    @property
    def platform(self):
        return self.pf

    @property
    def extrinsics_dirty(self):
        """ The extrinsics dirty signal """
        return self._extrinsics_dirty

    def invalidate_datasources_caches(self):
        """ Invalidate sensor's datasources caches """
        for ds in self.datasources.values():
            ds.invalidate_caches()
            
    def add_datasource(self, ds, ds_type:str, cache_size:int=100):
        """Adds a datasouce to this sensor

        Args:
            ds: a Filesource-derived instance or VirtualDatasource
            ds_type: the datasource type, e.g. 'ech'
        """

        if isinstance(ds, VirtualDatasource):
            ds._set_sensor(self)
            self.datasources[ds_type] = ds
        
        else:
            ds_type_no_suffix = ds_type.split('-')[0]
            dsw = DatasourceWrapper(self, ds_type, ds, self.factories[ds_type_no_suffix] if ds_type_no_suffix in self.factories else (Sample, nearest_interpolator), cache_size=cache_size)
            self.datasources[ds_type] = dsw

    def load_intrinsics(self, intrinsics_config:str):
        """Looks for a pickle file containing intrinsics information for this sensor, e.g. 'eagle_tfc.pkl'

        Args:
            intrinsics_config: path to folder containing this sensor's intrinsics pickle file, 
            (absolute or relative to dataset path), e.g. '/nas/cam_intrinsics' or 'cam_intrinsics'
        """
        
        paths = glob.glob(os.path.join(self.pf.try_absolute_or_relative(intrinsics_config), '{}*results.pkl'.format(self.name)))
        if paths:
            path = paths[0]
            if len(paths) > 1:
                LoggingManager.instance().warning('more than one intrinsics, using {}'.format(path))
            with open(path, 'rb') as f:
                self.intrinsics = pickle.load(f)

    def load_extrinsics(self, extrinsics_folder:str):
        """Looks for a pickle file containing extrinsics information for this sensor, named 'From-To' e.g. 'flir_tfl-eagle_tfc.pkl'

        Args:
            intrinsics_config: path to folder containing this sensor's extrinsics pickle file 
            (absolute or relative to dataset path), e.g. '/nas/extrinsics' or 'extrinsics'
        """
        targets = {}

        for target in self.pf.yml.keys():
            if self.name == target:
                targets[target] = np.eye(4, dtype = 'f8')
                continue

            # try to find self.name -> target mapping
            extrinsics_folder_path = self.pf.try_absolute_or_relative(extrinsics_folder)

            paths = glob.glob(os.path.join(extrinsics_folder_path, f"{self.name}-{target}.pkl"))

            if paths:
                with open(paths[0], 'rb') as f:
                    targets[target] = pickle.load(f).astype('f8')
            else:
                # try to find target -> self.name mapping instead
                paths = glob.glob(os.path.join(extrinsics_folder_path, f"{target}-{self.name}.pkl"))
                if paths:
                    with open(paths[0], 'rb') as f:
                        targets[target] = linalg.tf_inv(pickle.load(f)).astype('f8')


        self.extrinsics = targets

    def create_egomotion_provider(self) -> Optional['EgomotionProvider']:
        return None

    def map_to(self, target:str) -> np.ndarray:
        """Returns a 4x4 tranform matrix mapping a point from this sensor's referential to 'target' sensor's referential
        
        Args:
            target: the name of the target sensor in which referential we want to map to
        Raises:
            Sensor.NoPathToReferential: if no mapping could be found
        """

        return self._map_to_recurse(target, self.name, set([self.name]))

    def _map_to_recurse(self, target, orig, visited):

        if target in self.extrinsics:
            return self.extrinsics[target]

        for alt_target,v in self.extrinsics.items():
            if alt_target in visited:
                continue
            try:
                visited.update([alt_target])
                m = self.pf.sensors[alt_target]._map_to_recurse(target, alt_target, visited)
                return np.matmul(m, v)
            except:
                continue # we will try another path...

        raise Sensor.NoPathToReferential(f'Could not find a way to project from {self.name} to {target}')


    def keys(self):
        """Returns the datasource types, implement dict API"""
        return self.datasources.keys()

    def items(self):
        """Returns the datasource key,value iterable, implement dict API"""
        return self.datasources.items()

    def __contains__(self, key:str):
        """Returns wether 'key' is one of this sensor's datasources, implements 'in' API.

        To use: 
        >> 'ech' in LCAx('eagle_tfc', None)
        >> True
        """
        return key in self.datasources

    def __len__(self):
        """Returns the number of datasources in this sensor. Implements len() API """
        return len(self.datasources)

    def __getitem__(self, key:str):
        """Returns the datasource

        Args:
            key: the datasource type, e.g. 'ech'
        """
        
        try:
            return self.datasources[key]
        except KeyError:
            raise KeyError('This data source for sensor {} does not exist.'.format(self.name))
