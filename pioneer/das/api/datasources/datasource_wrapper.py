from pioneer.common import platform
from pioneer.das.api.interpolators import from_float_index
from pioneer.das.api.datasources import AbstractDatasource
from pioneer.das.api.sources import FileSource

from collections import deque
from typing import Optional, List, Callable, Dict, Tuple, Any, Iterable, Union

import numbers
import numpy as np

class DatasourceWrapper(AbstractDatasource):

    def __init__(self, sensor:'Sensor', ds_type:str, ds:FileSource, sample_interp:Tuple[Any, Any], cache_size:int=100):
        super(DatasourceWrapper, self).__init__(sensor, ds_type)
        self.sensor = sensor
        self.ds = ds
        self.ds_type = ds_type
        self._cache = {}
        self._cache_keys = deque(maxlen = cache_size)
        self.sample_factory, self.interpolator = sample_interp
        self.use_cache = cache_size > 0
        
    def invalidate_caches(self):
        self._cache = {}
        self._cache_keys = deque(maxlen = self._cache_keys.maxlen)
        self._interpolated = {}
        self._interpolated_keys = deque(maxlen = self._cache_keys.maxlen)

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, interpolator):
        self._interpolator = interpolator
        self._interpolated = {}
        self._interpolated_keys = deque(maxlen = self._cache_keys.maxlen)

    @property
    def timestamps(self):
        return self.ds.timestamps

    @property
    def time_of_issues(self):
        return self.ds.time_of_issues

    def _add_interpolated_from_float_index(self, float_index:float):
        if self.interpolator is None:
            raise RuntimeError('Index is float {} : No inperpolator found'.format(float_index))
        raw = self.interpolator(self, float_index)
        self._interpolated_keys.append(float_index)
        v =  self._interpolated[float_index] = self.sample_factory(float_index, self.sensor[self.ds_type], raw, self.to_timestamp(float_index))
        return v

    def to_timestamp(self, float_index:float) -> int:
        i, t = from_float_index(float_index)
        ts_from = self.timestamps[i - 1]
        return int(ts_from + t * (self.timestamps[i] - ts_from))

    def _add_interpolated_from_timestamp(self, timestamp:int) -> float:
        float_index = self.to_float_index(timestamp)
        self._add_interpolated_from_float_index(float_index)
        return float_index

    def get_at_timestamp(self, timestamp:Union[int, Iterable], interpolator:Callable=None) -> Union['Sample', List['Sample']]:
        try: #is timestamp iterable?
            return [self.get_at_timestamp(t, interpolator) for t in timestamp]
        except:
            float_index = self.to_float_index(timestamp)
            if interpolator is None:
                return self[float_index]
            return self.sample_factory(float_index, self.sensor[self.ds_type], interpolator(self, float_index), timestamp)

    def __getitem__(self, key:Any) -> Union['Sample', List['Sample']]:

        def try_get_from_cache_or_make_room(i, cache, cache_keys):
            if not self.use_cache:
                return None
            n = len(cache_keys)
            if i not in cache_keys:
                if n >= cache_keys.maxlen:
                    k = cache_keys.popleft()
                    s = cache.pop(k)
                    del s
                return None
            if n > 1:
                cache_keys.append(cache_keys.popleft()) #make latest access the most recent
            return cache[i]

        def instantiate(i):

            i = self.to_positive_index(i)
               
            if i >= len(self):
                raise IndexError("Out of bound")
            
            v = try_get_from_cache_or_make_room(i, self._cache, self._cache_keys)

            if v is None:
                if self.use_cache:
                    self._cache_keys.append(i)
                    v = self._cache[i] = self.sample_factory(i, self.sensor[self.ds_type])
                else:
                    v = self.sample_factory(i, self.sensor[self.ds_type])
                if self.sensor.platform.is_live():
                    # When the platform is live the fifo could fill up before data is accessed, 
                    # so we must avoid relying on it to get Sample.raw, Sample.timestamp
                    v.virtual_raw = v.raw
                    v.virtual_ts = v.timestamp

            return v

            self._cache_keys.append(self._cache_keys.popleft()) #make latest access the most recent
            return self._cache[i]

        try: #is key iterable?
            return [self[index] for index in key]
        except:
            if isinstance(key, slice):
                return self[platform.slice_to_range(key, len(self))]
            elif isinstance(key, numbers.Integral):
                return instantiate(key)
            elif isinstance(key, numbers.Real) and float(key).is_integer():
                return instantiate(int(key))
            elif isinstance(key, numbers.Real) and not float(key).is_integer():
                v = try_get_from_cache_or_make_room(key, self._interpolated, self._interpolated_keys)
                if v is None:
                    v = self._add_interpolated_from_float_index(key)

                return v

        raise KeyError('Invalid key: {}'.format(key))

    @property
    def label(self):
        return "{}_{}".format(self.sensor.name, self.ds_type)

    def get_timestamp_slice(self, timestamp:int, half_window_size:Union[int,tuple]):
        """Get an index slice that represent a time interval.

        The time interval is parametrized by a timestamp and a half window
        size.

        Args:
            timestamp: The center of the time interval
            half_window_size: Half of the time interval. If tuple provided -> (start, end).

        Returns:
            slice -- The indices slice representing all the elements that fit
            in the time interval
        """
        try: #is 2-tuple?
            left = int(timestamp) + int(half_window_size[0])
            right = int(timestamp) + int(half_window_size[1])
        except:
            left = int(timestamp) - int(half_window_size)
            right = int(timestamp) + int(half_window_size)
        left_index = int(np.searchsorted(self.timestamps, left, side='left'))
        right_index = int(np.searchsorted(self.timestamps, right, side='right'))
        return slice(left_index, right_index, 1)

