from pioneer.das.api.samples.sample import Sample

from typing import Callable

import copy
import numpy as np


class Trace(Sample):
    """Trace (or waveform) data from a single LCAx package"""

    def __init__(self, index, datasource, virtual_raw=None, virtual_ts=None):
        super(Trace, self).__init__(index, datasource, virtual_raw, virtual_ts)

    @property
    def raw(self):
        if self._raw is None:
            self._raw = super(Trace, self).raw

            for attribute in ['time_base_delays', 'distance_scaling', 'trace_smoothing_kernel']:
                if attribute not in self._raw and hasattr(self.datasource.sensor, attribute):
                    self._raw[attribute] = getattr(
                        self.datasource.sensor, attribute)

        return self._raw

    @property
    def raw_array(self):
        specs = self.specs
        raw = self.raw['data']
        return raw.reshape(specs['v'], specs['h'], raw.shape[-1])

    def processed_array(self, trace_processing: Callable):
        specs = self.specs
        processed = self.processed(trace_processing)['data']
        return processed.reshape(specs['v'], specs['h'], processed.shape[-1])

    @property
    def specs(self):
        # override the sensor specs if they are present in the YAML config file
        sensor_specs = self.datasource.sensor.specs
        if sensor_specs is not None:
            return sensor_specs
        return {k: self.raw[k] for k in ['v', 'h', 'v_fov', 'h_fov']}

    @property
    def timestamps(self):
        return self.raw['t']

    def processed(self, trace_processing: Callable):
        raw_copy = copy.deepcopy(self.raw)
        processed_traces = trace_processing(raw_copy)
        return processed_traces

    @staticmethod
    def saturation_flags(traces):
        flags = np.zeros(traces['data'].shape[0], dtype='u2')

        traces['data'] = traces['data'].astype('float64')
        saturation_value = traces['data'].max()
        if saturation_value == 0:
            return traces

        where_plateau = np.where(traces['data'] == saturation_value)
        channels, ind, sizes = np.unique(
            where_plateau[0], return_index=True, return_counts=True)
        positions = where_plateau[1][ind]

        for channel, position, size in zip(channels, positions, sizes):
            if size > 5 and position > 2 and position + size + 2 < traces['data'].shape[-1]:
                flags[channel] = 1
        return flags

    @property
    def max_range(self):
        raw = self.raw
        trace_lenght = raw['data'].shape[-1]
        return raw['time_base_delays'] + trace_lenght*raw['distance_scaling']

    @property
    def signal_to_noise(self):
        traces_zeroed = self.raw['data'] - \
            np.mean(self.raw['data'], axis=1)[:, None]
        return np.log10(np.mean(traces_zeroed**2, axis=1)/np.std(traces_zeroed, axis=1))+1
