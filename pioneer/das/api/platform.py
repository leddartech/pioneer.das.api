from pioneer.common import platform as platform_utils
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.datasources import AbstractDatasource
from pioneer.das.api.datasources import virtual_datasources
from pioneer.das.api.samples.sample import Sample
from pioneer.das.api.sensors import Sensor
from pioneer.das.api.sensors import SENSOR_FACTORY
from pioneer.das.api.sources import DirSource, ZipFileSource

from typing import Callable, Iterator, Union, Optional, List, Dict, Mapping, Tuple, Any

import ast
import glob
import numbers
import numpy as np
import os
import pandas as pd
import six
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import sys
import tqdm
import yaml
import zipfile

def parse_yaml_string(ys):
    fd = StringIO(ys)
    dct = yaml.safe_load(fd)
    return dct
    
try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    LoggingManager.instance().warning('Numba is not installed, you may have MemoryErrors when '
                  'synchronizing data sources.')
    HAVE_NUMBA = False

def closest_timestamps_np(ref_ts:np.ndarray, target_ts:np.ndarray, tol:Union[float, int]):
    """Finds indices of timestamps pairs where a value in 'target_ts' is whithin 'tol' of a value 'ref_ts'

        Args:
            ref_ts: the timestamps of the sensor we want to synchronize with
            target_ts: the timestamps of the sensor we hope to find matches with 'ref_ts' whithin 'tol'
        Returns:
            The indices of the matches
    """
    try:
        diff = np.abs(ref_ts[:, None].astype('i8') -
                        target_ts[None, :].astype('i8'))
    except MemoryError:
        print("You don't have enough memory. "
              "Try installing numba to resolve this issue.\n"
              "pip3 install numba or conda install numba",
              file=sys.sterr
              )
        raise

    # find the closest leddar timestamp for each camera timestamp
    idx = np.argmin(diff, axis=1)

    # what if many images match the same leddar data package equally?
    min_diff = diff[np.arange(diff.shape[0]), idx]
    too_large = min_diff > tol
    idx[too_large] = -1
    return idx

closest_timestamps = closest_timestamps_np

if HAVE_NUMBA:
    @numba.njit
    def closest_timestamps_numba(ref_ts, target_ts, tol):
        """ Numba-optimized version of closest_timestamps_np()"""
        n_ref = ref_ts.shape[0]
        ref_ts = ref_ts.astype(np.int64)
        target_ts = target_ts.astype(np.int64)
        indices = np.full(n_ref, -1)

        for i_ref in range(n_ref):
            diff = np.abs(ref_ts[i_ref] - target_ts)
            min_idx = np.argmin(diff)
            if diff[min_idx] > tol:
                continue
            indices[i_ref] = min_idx

        return indices
        
    closest_timestamps = closest_timestamps_numba

class Platform(object):
    """A Platform is what encapsulate an instance (configuration) of some data acquisition platform. 
       It contains one or more sensors, each containing one or more datasources. A live platform interfaces
       live sensors, while an offline platform can be used to extract data from a recording. 
    """

    def __init__(self, dataset:Optional[str]=None, configuration:Optional[str]=None, include:Optional[list]=None, ignore:Optional[list]=[], progress_bar:bool=True, default_cache_size:int=100):
        """Constructor

           Args:
            dataset:    If None, this platform will be considered a live sensor platform. 
                        Otherwise, 'dataset' contains the path to an offline recording, where 
                        one expects to find one or many '.zip' filesources named like their 
                        corresponding datasource. If argument 'configuration' is None, then one
                        expect to find a file named 'platform.yml' which declares which sensor this
                        platform must consider, and thus which '.zip' file to use. 'zip' files are 
                        expected to contain one file name according to das.api.filesource.TIMESTAMPS_CSV_PATTERN, 
                        zero or one file name according to das.api.filesource.CONFIG_YML_PATTERN, and 
                        other files corresponding to actual sensor data (e.g. '.pkl' files, '.png' files, etc)
            configuration:  If None, see 'dataset' documentation. Otherwise, 'configuration' is expected to contain
                            a path to a '.yml' file describing a das.api acquisition platform configuration, or its content
                            directly, which allows the user to instanciate a platform without creating a file. 
                            If 'dataset' is not None, 'configuration' can be a path relative to 'dataset'
            include:    A list of strings (e.g. ['lca2','flir_tfc']) to be included exclusively in the platform's keys
            ignore:     A list of strings (e.g. ['lca2','flir_tfc']) to be ignored and not included in the platform's keys
            progress_bar:   If True, show the progress bars when initializing or synchronizing.
            default_cache_size: the default value for (offline) datasource cache size
        """

        self.dataset = dataset
        self.progress_bar = progress_bar
        self.default_cache_size = default_cache_size
        
        self.configuration = configuration
        if self.configuration is None:
            self.configuration = glob.glob(os.path.join(dataset, 'platform.yml'))[0]

        is_config_a_file = os.path.exists(self.configuration) or os.path.isabs(self.configuration)

        if not is_config_a_file and self.dataset is not None:
            candidate = os.path.join(self.dataset, self.configuration)
            is_config_a_file = os.path.exists(candidate) or os.path.isabs(candidate)
            if is_config_a_file:
                self.configuration = candidate

        if is_config_a_file:
            with open(self.configuration, 'r') as f:
                self.yml = yaml.safe_load(f)
        else:
            self.yml = parse_yaml_string(self.configuration)

        self.include = include
        self.ignore = ignore

        # If there is a way to use wildcards here instead, it would be better
        include = self.yml.keys() if include is None else include
        to_keep, to_pop = [], []
        for additionnal_key in ['virtual_datasources', 'synchronization']:
            if additionnal_key in self.yml:
                to_keep.append(additionnal_key)
        for sensor in self.yml:
            for inc in include:
                if inc in sensor:
                    to_keep.append(sensor)
            for ign in ignore:
                if ign in sensor:
                    to_pop.append(sensor)
        self.yml = {sensor:self.yml[sensor] for sensor in to_keep}
        for sensor in set(to_pop):
            try:
                self.yml.pop(sensor)
            except:
                pass

        self.metadata_path = f'{dataset}/metadata.csv'
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path, index_col=0, converters={'keywords':ast.literal_eval})
            self.metadata = self.metadata.where(pd.notnull(self.metadata), None)
        else:
            self.metadata = None

        self._sensors = Sensors(self, self.yml)

        if 'virtual_datasources' in self.yml:
            self.add_virtual_datasources(self.yml['virtual_datasources'])

    def to_nas_path(self, path:str) -> str:
        """Convert absolute yaml paths to path relative to os.environ['nas'] """
        nas_base = os.environ.get('nas', '')
        return nas_base + path # do not use os.path.join() here, as nas defaults to /nas, which is absolute

    def from_relative_path(self, relative_path:str) -> str:
        """Converts a path relative to dataset folder to an absolute path """

        if self.dataset is None:
            raise RuntimeError("dataset path is None")

        return os.path.join(self.dataset, relative_path)

    def try_absolute_or_relative(self, folder):
        """ tries for absolute or relative path """

        folder_path = self.to_nas_path(folder)

        if not os.path.exists(folder_path):
            folder_path = self.from_relative_path(folder)

        if not os.path.exists(folder_path):
            raise RuntimeError(f"Invalid extrinsics path: {folder}")

        return folder_path

    @property
    def sensors(self) -> 'Sensors':
        """Returns this platform's Sensors instance"""
        return self._sensors

    @property
    def orientation(self) -> Dict[str, Optional[np.ndarray]]:
        """Returns a dict with orientation matrix for each platform's sensor"""
        return {sensor.name: sensor.orientation for sensor in self._sensors.values()}

    @property
    def intrinsics(self) -> Dict[str, Any]:
        """Returns a dict with intrinsics information for each platform's sensor"""
        return {sensor.name: sensor.intrinsics for sensor in self._sensors.values()}

    @property
    def extrinsics(self) -> Dict[str, Mapping[str, np.ndarray]]:
        """Returns a dict with extrinsics information for each platform's sensor"""
        return {sensor.name: sensor.extrinsics for sensor in self._sensors.values()}

    def expand_wildcards(self, labels:List[str]) -> List[str]:
        """See also: platform.expand_widlcards()
        """ 
        return platform_utils.expand_wildcards(labels, self.datasource_names())

    def synchronized(self, sync_labels:List[str]=[], interp_labels:List[str]=[], tolerance_us:Union[float, int]=None, fifo:int=-1):
        """Creates a Synchronized instance with self as platform
           
           See Also: Synchronized.__init__()
        """
        if 'synchronization' in self.yml:
            sync_labels = self.yml['synchronization']['sync_labels'] if len(sync_labels)==0 else sync_labels
            interp_labels = self.yml['synchronization']['interp_labels'] if len(interp_labels)==0 else interp_labels
            tolerance_us = self.yml['synchronization']['tolerance_us'] if tolerance_us is None else tolerance_us
        tolerance_us = 1e3 if tolerance_us is None else tolerance_us
        return Synchronized(self, self.expand_wildcards(sync_labels)
                            , self.expand_wildcards(interp_labels)
                            , tolerance_us, fifo)

    def is_live(self) -> bool:
        """Returns wether this platform contains live sensors or offline recordings"""
        return self.dataset is None

    def start(self):
        """**Live platform only** starts the (live) sensors"""
        self._sensors.start()

    def stop(self):
        """**Live platform only** stops the (live) sensors"""
        self._sensors.stop()
    
    def record(self):
        """**Live platform only** toggle recording for all (live) sensors"""
        for ds_name in self.datasource_names():
            try:
                self[ds_name].ds.is_recording = not self[ds_name].ds.is_recording
            except: pass
    
    @property
    def egomotion_provider(self) -> 'EgomotionProvider':
        return self._sensors._egomotion_provider

    def datasource_names(self) -> List[str]:
        """Returns the list of all datasource names, e.g. ['pixell_tfc_ech', 'pixell_tfc_sta', ...]"""
        ds_names = []
        for s in self._sensors.values():
            ds_names.extend(s.datasource_names())
        return ds_names

    def add_virtual_datasources(self, virtual_datasources_config):
        """Add virtual datasources based upon the provided configuration dictionnary
        
            Args:
                -virtual_datasources_config (dict): Each key must correspond to a key available in
                    VIRTUAL_DATASOURCE_FACTORY (see api/datasources/virtual_datasources/__init__.py).
                    Under each key, a dictionnary must be provided with the parameters necessary
                    to create the corresponding virtual datasource. These parameters always contain at 
                    least the 'reference_sensor' and 'dependencies' entries. Other entries depend upon
                    the specific virtual datasource type.

                    For exemple:
                        virtual_datasources_config = {
                            'echoes_from_traces': {
                                'reference_sensor': 'pixell_bfc',
                                'dependencies': ['pixell_bfc_ftrr'],
                                'nb_detections_max': 3,
                            }
                        }

            Note: To create multiple virtual datasources of the same type, you can do something like:
                virtual_datasources_config = {
                    'echoes_from_traces_bfc': {'reference_sensor': 'pixell_bfc', 'dependencies': ['pixell_bfc_ftrr']}
                    'echoes_from_traces_tfc': {'reference_sensor': 'pixell_tfc', 'dependencies': ['pixell_tfc_ftrr']}
                }
        """

        for virtual_ds_name in virtual_datasources_config:
            args = virtual_datasources_config[virtual_ds_name]

            # If multiple instances of the same VirtualDatasource class, their keys has to be different
            # in the config file. So add a unique id such as: virtual_ds_name -> virtual_ds_name_id
            if hasattr(virtual_datasources, virtual_ds_name[:virtual_ds_name.rfind('_')]):
                virtual_ds_name = virtual_ds_name[:virtual_ds_name.rfind('_')]

            if hasattr(virtual_datasources, virtual_ds_name):
                try:
                    virtual_datasource = virtual_datasources.VIRTUAL_DATASOURCE_FACTORY[virtual_ds_name](**args)
                    self[virtual_datasource.reference_sensor].add_datasource(virtual_datasource, virtual_datasource.ds_type)
                except:
                    LoggingManager.instance().warning(f"The virtual datasource {virtual_ds_name} could not be added.")
            else:
                LoggingManager.instance().warning(f"The virtual datasource {virtual_ds_name} does not exist.")
        
    def __getitem__(self, label:str) -> Union[Sensor, AbstractDatasource]:
        """Implement operator '[]'

           Args:
            label: can be a sensor name (e.g. 'pixell_tfc') or a datasource name (e.g. 'pixell_tfc_ech')
           Returns:
            the Sensor or AbstractDatasource instance
           Raises:
            IndexError: label must contain exactly 2 or 3 '_' 
        """
        parts = label.split('_')
        if len(parts) == 2:
            return self.sensors[label]
        if len(parts) == 3:
            return self.sensors[parts[0] + '_' + parts[1]][parts[2]]
        else:
            raise IndexError('Invalid label: {}'.format(label))


class Synchronized(object):
    """Synchronized view over a Platform"""

    def __init__(self, platform:Platform, sync_labels:List[str], interp_labels:List[str], tolerance_us:Union[float, int], fifo:int):
        """Constructor

            Args:
                platform:       The platform that contains datasources to synchronize
                sync_labels:    A list of expanded datasource names to harvest tuples of samples synchronized 
                                whithin 'tolerance_us' microseconds. **Important** The first label will serve as the 'reference 
                                datasourse'
                interp_labels:  A list of expanded datasource names from which to obtain interpolated samples 
                                to be added to the 'synchronized tuples'. 
                                The corresponding datasource must have an interpolator defined.
                tolerance_us:   the synchronization tolerance, i.e. for each sample in 'reference datasource', we try to find one sample
                                in each of the other synchronized datasources that is whitin 'tolerance_us' and add it to the 'synchronized tuple'.
                                Incomplete 'synchronized tuples' are discarded.
                fifo:           For future use (live datasources)
        """
        self.platform = platform
        self.sync_labels = sync_labels
        self.interp_labels = interp_labels
        self.tolerance_us = tolerance_us
        self.fifo = fifo
        self.sync_indices = None
        self.ref_index = 0

        if not self.platform.is_live():
            self.mappings = self._synchronize_offline_sensor_data()
            self.sync_indices = self.mappings[(self.mappings != -1).all(axis=1), :]

    @property
    def ref_ds_name(self):
        return self.sync_labels[self.ref_index]

    def _synchronize_offline_sensor_data(self):
        """Synchronizes all datasources in the sync_labels list together. The first 
           datasource in the list will be considered the 'reference datasource' 
        """
        self.all_ts = [(c, self.platform[c].timestamps) for c in self.sync_labels]
        self.ref_ts = dict(self.all_ts)[self.ref_ds_name]
        mappings = []

        all_ts_tqdm = tqdm.tqdm(self.all_ts, 'Synchronizing') if self.platform.progress_bar else self.all_ts
        for name, sensor_ts in all_ts_tqdm:
            
            if sensor_ts.size == 0:
                raise RuntimeError('Sensor {} has 0 timestamps, could not \
                synchronize arrays, please exclude this label from list'
                .format(name))
            
            idx = closest_timestamps(self.ref_ts, sensor_ts, self.tolerance_us)
            mappings.append(idx)

        return np.stack(mappings, axis=1)

    def expand_wildcards(self, labels:List[str]):
        """See also: platform.expand_widlcards()""" 
        return platform_utils.expand_wildcards(labels, self.keys())

    def keys(self):
        """Returns synchronization labels and interpolation labels (implement dict API)"""
        return self.sync_labels + self.interp_labels

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return self.sync_indices.shape[0] if self.sync_indices is not None else 0

    def get_single_ds(self, ds_name:str, index:Union[int, slice, Iterator[int]]) -> Sample:

        try: #is index iterable?
            l = []
            for i in index:
                l.append(self.get_single_ds(ds_name, i))
            return l
        except:
            if isinstance(index, slice):
                return self.get_single_ds(ds_name, platform_utils.slice_to_range(index, len(self)))
            elif isinstance(index, numbers.Integral):
                data_indices = self.sync_indices[index]
                try:
                    i_ds = self.sync_labels.index(ds_name)

                except:
                    try:
                        i_ds = self.interp_labels.index(ds_name)
                    except:
                        raise RuntimeError(f"Unexpected label {ds_name}")
                    return self.platform[ds_name].get_at_timestamp(self.ref_ts[data_indices[self.ref_index]])

                return self.platform[ds_name][data_indices[i_ds]]

            raise KeyError("Unsupported key type: {}".format(index))
    
    class SynchGetter(object):
        def __init__(self, index, synch):
            self.index = index
            self.synch = synch
            
        def __getitem__(self, ds_name):
            return self.synch.get_single_ds(ds_name, self.index)

        def __setitem__(self, key, item):
            raise RuntimeError("Immutable")

        def __delitem__(self, key):
            raise RuntimeError("Immutable")

        def __repr__(self):
            return repr(self.items())
        
        def keys(self):
            return self.synch.keys()

        def clear(self):
            raise RuntimeError("Immutable")

        def copy(self):
            return Synchronized.SynchGetter(self.index, self.synch)

        def values(self):
            return [self[k] for k in self.keys()]
        
        def items(self):
            return [(k, self[k]) for k in self.keys()]

        def len(self):
            return len(self.keys())

        def has_key(self, k):
            return k in self.keys()
        
        def __contains__(self, item):
            return item in self.values()

        def __iter__(self):
            return iter(self.values())

    def __getitem__(self, index:Union[int, slice, Iterator[int]]) ->Dict[str, Sample]:
        """Implements the '[]' API

            Args:
                index: the (complete) datasource name
            Returns:
                A Sample

        """
        
        return Synchronized.SynchGetter(index, self)

    def timestamps(self, index:int) -> np.ndarray:
        """Returns timestamps for all synchronized datasources for a given index"""
        ts = []
        for source_i, sample_i in enumerate(self.sync_indices[index, :]):
            source = self.platform[self.sync_labels[source_i]]
            ts.append(source.timestamps[sample_i])
        return np.array(ts)

    def indices(self, index:int):
        """Returns indices for all datasources for a given synchronized index"""
        interp_indices = []
        for ds in self.interp_labels:
            interp_indices.append(self.platform[ds].to_float_index(self.ref_ts[self.sync_indices[index][self.ref_index]]))
        return list(self.sync_indices[index]) + interp_indices

    def sliced(self, intervals:List[Tuple[int, int]]):
        """Returns a Sliced instance
        
        See Also: Sliced
        """
        return Sliced(self, intervals)

    def filtered(self, indices:List[int]):
        return Filtered(self, indices)


class SynchronizedGroup(Synchronized):
    """Groups multiple synchronized platforms in a single one
        Args:
            -datasets: Can be either a list of paths for the datasets to be grouped, 
                or a single string for the path of a directory that contains the datasets that will all be grouped.
            -sync_labels: see Synchronized
            -interp_labels: see Synchronized
            -tolerance_us: see Synchronized
            -include: see Platform
            -ignore: see Platform
            -preload: If True, all platforms will be initialized and synchronized at start. 
                Good for random or non-sequential data access, but initialization is slower.
    """

    def __init__(self, datasets:Union[str,list], sync_labels:List[str]=[], interp_labels:List[str]=[], tolerance_us:Union[float, int]=None,
                     include:Optional[list]=None, ignore:Optional[list]=[], preload:bool=False, virtual_datasources_config:dict=None):
                     
        if type(datasets) == str:
            datasets = glob.glob(f"{datasets}/*")

        self.datasets = datasets
        self.sync_labels = sync_labels
        self.interp_labels = interp_labels
        self.tolerance_us = tolerance_us
        self.include = include
        self.ignore = ignore
        self.preload = preload
        self.virtual_datasources_config = virtual_datasources_config

        self.cached_synchronized = [None for d in self.datasets]
       
        # The first platform is synchronized at start
        sync0 = self.get_synchronized(0, progress_bar=True)
        self.sync_labels = sync0.sync_labels
        self.interp_labels = sync0.interp_labels
        self.tolerance_us = sync0.tolerance_us

        self._preload_lenghts()

    def _preload_lenghts(self):
    
        self.lenghts = [] 
        for dataset_index in tqdm.tqdm(range(len(self.datasets)), 'Grouping synchronized platforms'):
            try:
                if self.preload:
                    l = self._get_dataset_lenght(dataset_index)
                else:
                    l = self._get_dataset_lenght_fast(dataset_index)
            except Exception:
                l = 0
                LoggingManager.instance().warning(f"The dataset {self.datasets[dataset_index]} could not be added to the SynchronizedGroup.")

            self.lenghts.append(l)

        self.cumsum_lenghts = np.cumsum([0]+self.lenghts[:-1])

    def get_synchronized(self, dataset_index, progress_bar=False):
        if self.cached_synchronized[dataset_index] is None:
            try:
                pf = Platform(self.datasets[dataset_index], include=self.include, ignore=self.ignore, progress_bar=progress_bar, default_cache_size=0)
                if self.virtual_datasources_config is not None:
                    pf.add_virtual_datasources(self.virtual_datasources_config)
                sync = pf.synchronized(self.sync_labels, self.interp_labels, self.tolerance_us)
                self.cached_synchronized[dataset_index] = sync
            except Exception:
                LoggingManager.instance().warning(f"There is an issue with the dataset {self.datasets[dataset_index]}.")
        return self.cached_synchronized[dataset_index]

    def _get_dataset_lenght(self, dataset_index):
        sync = self.get_synchronized(dataset_index)
        if not all([ds in sync.keys() for ds in self.sync_labels]):
            return 0
        return len(sync)

    def _get_dataset_lenght_fast(self, dataset_index):
        """
        This is a faster way to get the lenght of a synchronized platform, without initalizing it.
        We directly load the timestamp.csv files and check how many matching frames are found.
        """
        # If the synchronized platform is already cached, get its lenght
        if self.cached_synchronized[dataset_index] is not None:
            return len(self.cached_synchronized[dataset_index])

        # Here, we determine for which datasources we have to load a timestamp.csv file
        datasources_with_ts = []
        for ds_name in self.sync_labels:
            datasource = self.cached_synchronized[0].platform[ds_name]
            if isinstance(datasource, virtual_datasources.VirtualDatasource):
                continue
            datasources_with_ts.append(f"{ds_name}")

        ref_ts = []
        sensor_ts = []
        mappings = []

        for ds_name in datasources_with_ts:

            # zipped case
            if os.path.exists(f"{self.datasets[dataset_index]}/{ds_name}.zip"):
                path = f"{self.datasets[dataset_index]}/{ds_name}.zip"
                with zipfile.ZipFile(path, 'r') as zf:
                    with zf.open('timestamps.csv') as stream:
                        ts = pd.read_csv(stream, delimiter=" ", dtype='u8', header=None).values[:,0]

            # unzipped case
            elif os.path.exists(f"{self.datasets[dataset_index]}/{ds_name}/timestamps.csv"):
                path = f"{self.datasets[dataset_index]}/{ds_name}/timestamps.csv"
                with open(path) as stream:
                    ts = pd.read_csv(stream, delimiter=" ", dtype='u8', header=None).values[:,0]
            else:
                return 0

            if ref_ts == []:
                ref_ts = ts

            idx = closest_timestamps(ref_ts, ts, self.tolerance_us)
            mappings.append(idx)
        
        mappings = np.stack(mappings, axis=1)
        mappings = mappings[(mappings != -1).all(axis=1), :]

        return mappings.shape[0]

    def __len__(self):
        return sum(self.lenghts)

    @property
    def platform(self):
        return self.cached_synchronized[0].platform
 
    def timestamps(self, index:int):
        raise RuntimeError('not implemented!')

    def indices(self, index:int):
        raise RuntimeError('not implemented!')

    def sliced(self, intervals:List[Tuple[int, int]]):
        raise RuntimeError('not implemented!')

    def filtered(self, indices:List[int]):
        raise RuntimeError('not implemented!')

    def keys(self):
        return self.cached_synchronized.keys()

    def _group_index(self, index:int):
        """For a given index from 0 to the summed lenght of all grouped datasets,
        finds in which dataset to get the samples and what is the corresponding index in this dataset.
        Ex: If we grouped two datasets of lenght 1000 each, then _group_index(1500) would return 
            the pair 1, 500. So, the samples are at index 500 in the second (dataset_index=1) dataset.
        """
        if index > len(self) or index < 0:
            raise IndexError("Index out of bounds")
        
        cumsum_index = index-self.cumsum_lenghts
        dataset_index = np.where(cumsum_index<self.lenghts)[0][0]
        idx = cumsum_index[dataset_index] #find the index in the specific dataset
        return dataset_index, idx

    def __getitem__(self, index:Union[int, slice, Iterator[int]]) ->Dict[str, Sample]:

        list_index = []
        try:
            for i in index:
                list_index.append(i)
        except:
            if isinstance(index, slice):
                list_index = list(platform_utils.slice_to_range(index, len(self)))
            if isinstance(index, numbers.Integral):
                list_index = [index]
        
        merged = None
        for i in list_index:
            dataset_index, idx = self._group_index(i)
            samples = self.get_synchronized(dataset_index)[idx]
            if merged is None:
                merged = {k:[v] for k,v in samples.items()}
            else:
                for k,v in merged.items():
                    v.append(samples[k])
        return merged



class Filtered(object):

    def __init__(self, synchronized:Synchronized, indices:List[int]):
        self.synchronized = synchronized
        self.indices = indices

    @property
    def platform(self):
        return self.synchronized.platform
        
    def expand_wildcards(self, labels:List[str]):
        """Wraps Synchronized.expand_widlcards()
        """ 
        return self.synchronized.expand_wildcards(labels)

    def keys(self):
        """Wraps Synchronized.keys()"""
        return self.synchronized.keys()

    def __contains__(self, key):
        """Wraps Synchronized.__contains__()"""
        return self.synchronized.__contains__(key)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index:Any) ->Dict[str, Sample]:
        """Implements the '[]' API, using the intervals

        Args:
            index: Will index in this view's intervals, and then call Synchronized.__getitem__() 
        Returns:
            See Synchronized.__getitem__()

        """
        return self.synchronized[self.indices[index]]
        

class Sliced(object):
    """Creates a view on a synchronized dataset using a list of intervals"""

    def __init__(self, synchronized:Synchronized, intervals:List[Tuple[int, int]], stride:int = 1):
        """Constructor

            Args:
                synchronized:   The 'Synchronized' instance we want a view on.
                intervals:      An list of (open, open) intervals of interest over synchronized's domain. 
                                The list does not need to be ordered, and can index a synchronized index more than once.
                                For example, with a 'stride' of 1, the list [(5, 7), (12, 15), (15, 13)] will expand to indices [5,6,7, 12,13,14,15, 15,14,13].
                stride:         the stride (jump between consecutive frames)
                              
        """
        self.synchronized = synchronized
        self.intervals = intervals
        self.stride = stride
        self.indices = []
        n = len(synchronized)
        for s,e in intervals:
            expanded = []
            if s>=0 and e>=0 and s > e: #inverted
                expanded = [i for i in range(e+1, s+1, self.stride)][::-1]
            else:
                expanded = [i for i in range(s, e+1, stride)]

            self.indices.extend(expanded)
    
    @property
    def platform(self):
        return self.synchronized.platform
        
    def expand_wildcards(self, labels:List[str]):
        """Wraps Synchronized.expand_widlcards()
        """ 
        return self.synchronized.expand_wildcards(labels)

    def keys(self):
        """Wraps Synchronized.keys()"""
        return self.synchronized.keys()

    def __contains__(self, key):
        """Wraps Synchronized.__contains__()"""
        return self.synchronized.__contains__(key)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index:Any) ->Dict[str, Sample]:
        """Implements the '[]' API, using the intervals

        Args:
            index: Will index in this view's intervals, and then call Synchronized.__getitem__() 
        Returns:
            See Synchronized.__getitem__()

        """
        return self.synchronized[self.indices[index]]


class Sensors(object):
    """The collection of Sensor instances in a platform"""
    def __init__(self, pf:'Platform', yml:dict):
        """Constructor

           Args:
            platform: the platform that holds this Sensors instance
            yml: the YAML database
        """
        self.platform = pf
        self._sensors = {}
        self._ordered_names = []
        self._egomotion_provider = None

        yml_sensor_items = [s for s in yml.items() if s[0] not in ['ignore','virtual_datasources','synchronization']]
        yml_items_tqdm = tqdm.tqdm(yml_sensor_items, 'Loading sensors') if self.platform.progress_bar else yml_sensor_items
        for name, value in yml_items_tqdm:

            sensor_type, _ = platform_utils.parse_sensor_name(name)
            if sensor_type in SENSOR_FACTORY:
                self._sensors[name] = SENSOR_FACTORY[sensor_type](name, self.platform)
            else:
                self._sensors[name] = Sensor(name, self.platform)
            self._ordered_names.append(name)

            self._load_offline_datasources(name)

            if 'orientation' in value:
                m = np.array(value['orientation'], dtype = 'f4')
                if m.shape != (3,3):
                    LoggingManager.instance().warning('Ignoring orientation for sensor {}: {} is \
                    not a 3 x 3'.format(name, str(m)))
                else:
                    self._sensors[name].orientation = m

            if 'intrinsics' in value:
                self._load_intrinsics(name, value['intrinsics'])

            if 'extrinsics' in value:
                self._load_extrinsics(name, value['extrinsics'])

            try:
                provider = self._sensors[name].create_egomotion_provider()
            except:
                LoggingManager.instance().warning(f"The 'egomotion_provider' for sensor name {name} could not be created.")

            if provider is not None:
                if self._egomotion_provider is not None:
                    LoggingManager.instance().warning(f"Another 'egomotion_provider' found for sensor name {name}, ignoring it.")
                else:
                    self._egomotion_provider = provider


    def start(self):
        """**Live platform only** starts the (live) sensors"""
        for name, sensor in self._sensors.items():
            sensor.start()

    def stop(self):
        """**Live platform only** starts the (live) sensors"""
        for name, sensor in self._sensors.items():
            sensor.stop()
    
    def _load_intrinsics(self, name, intrinsics_config):
        """Intrinsics config can be a string or a dict so complete the path,
           only if the intrinsics config is a string.
        """
        if isinstance(intrinsics_config, six.string_types):
            intrinsics_config = self.platform.to_nas_path(intrinsics_config)
        self._sensors[name].load_intrinsics(intrinsics_config)

    def _load_extrinsics(self, name, extrinsics_folder):
        """ Extrinsics files are pkl files that contain a 4x4 numpy array that
            allows to project 3d point(s) form a sensor's referential to another
            sensor's referential. The expected file format is
            "{}-{}.pkl".format(source_sensor_name, destination_sensor_name).
            For example, file "eagle_bcc-flir_tfl.pkl" would contain a 4x4 numpy
            array that allows to project an eagle_bcc's point cloud in flir_tfl's
            referential.

            Note it will also try to find
            "{}-{}.pkl".format(destination_sensor_name, source_sensor_name) and
            use that transform's inverse
        """
        self._sensors[name].load_extrinsics(extrinsics_folder)

    def _load_offline_datasources(self, name:str):

        archives = glob.glob(os.path.join(self.platform.dataset, name + '_*'))
        for archive in [os.path.basename(f) for f in archives]:

            if archive.endswith('.zip'):
                # remove the .zip and extracts the 'datasource' suffix:
                ds_name = os.path.splitext(archive)[0].split('_')[-1]#.split('-')[0]
                try:
                    ds = ZipFileSource(os.path.join(self.platform.dataset, archive))
                    self._sensors[name].add_datasource(ds, ds_name, cache_size = self.platform.default_cache_size)
                except:
                    LoggingManager.instance().warning(f'Zip file for {name}_{ds_name} could not be loaded.')
                    continue

            else:
                ds_name = archive.split('_')[-1]#.split('-')[0]
                try:
                    ds = DirSource(os.path.join(self.platform.dataset, archive))
                    self._sensors[name].add_datasource(ds, ds_name, cache_size = self.platform.default_cache_size)
                except:
                    LoggingManager.instance().warning(f'Data directory for {name}_{ds_name} could not be loaded.')
                    continue

    def override_sensor_extrinsics(self, name:str, extrinsics_folder:str):
        """Override the extrinsics that we loaded from the platfrom yml.

        Args:
            extrinsics_folder: The extrinsics folder
        """
        self._load_extrinsics(name, extrinsics_folder)

    def override_all_extrinsics(self, extrinsics_folder:str):
        """Override the extrinsics that we loaded from the platfrom yml.

        Only the sensor for which the extrinsics were previously loaded
        will be overriden.

        Args:
            extrinsics_folder: The extrinsics folder
        """
        for name, sensor in self.items():
            if sensor.extrinsics is not None:
                self._load_extrinsics(name, extrinsics_folder)

    def keys(self):
        """Implement dict API
            Returns:
                The list of sensor names, e.g. ['pixell_tfc', 'flir_tfc', ...]
        """
        return self._sensors.keys()

    def items(self):
        """Implement dict API"""
        return self._sensors.items()

    def values(self):
        """Implement dict API"""
        return self._sensors.values()

    def __contains__(self, key):
        return key in self._sensors

    def __len__(self):
        return len(self._sensors)

    def __getitem__(self, label):
        """Implement '[]' API"""
        return self._sensors[label]

    def __iter__(self):
        return iter(self._sensors)