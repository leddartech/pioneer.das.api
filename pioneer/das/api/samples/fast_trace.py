from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.samples.trace import Trace

from typing import Callable

import copy
import numpy as np


class FastTrace(Trace):
    """Derivation of Trace sample, for Pixell sensors. 
        Two distinct sets of waveforms are contained in the raw data dictionnary, under the keys 'high' and 'low'.
    """

    def __init__(self, index, datasource, virtual_raw=None, virtual_ts=None):
        super(FastTrace, self).__init__(
            index, datasource, virtual_raw, virtual_ts)

    @property
    def raw(self):
        if self._raw is None:
            self._raw = super(Trace, self).raw

            for fast_trace_type in ['low', 'high']:
                self._raw[fast_trace_type]['time_base_delays'] = self.datasource.sensor.time_base_delays[fast_trace_type]
                self._raw[fast_trace_type]['distance_scaling'] = self.datasource.sensor.distance_scaling
                self._raw[fast_trace_type]['trace_smoothing_kernel'] = self.datasource.sensor.get_trace_smoothing_kernel()[
                    fast_trace_type]

        return self._raw

    @property
    def raw_array(self):
        specs = self.specs
        rawLow = self.raw['low']['data']
        rawMid = self.raw['high']['data']
        array = np.zeros((2, specs['v']*specs['h'], rawMid.shape[-1]))
        array[0] = rawMid
        array[1, :, :rawLow.shape[-1]] = rawLow
        return array.reshape((2, specs['v'], specs['h'], array.shape[-1]))

    def processed_array(self, trace_processing: Callable):
        specs = self.specs
        processedLow = self.processed(trace_processing)['low']['data']
        processedMid = self.processed(trace_processing)['high']['data']
        array = np.zeros((2, specs['v']*specs['h'], processedMid.shape[-1]))
        array[0] = processedMid
        array[1, :, :processedLow.shape[-1]] = processedLow
        return array.reshape((2, specs['v'], specs['h'], array.shape[-1]))

    def processed(self, trace_processing: Callable):
        processed_traces = {}
        raw_copy = copy.deepcopy(self.raw)
        for fast_trace_type in ['low', 'high']:
            processed_traces[fast_trace_type] = trace_processing(
                raw_copy[fast_trace_type])
        return processed_traces

    @property
    def max_range(self):
        raw = self.raw
        trace_lenght = raw['high']['data'].shape[-1]
        return raw['high']['time_base_delays'] + trace_lenght*raw['high']['distance_scaling']

    @property
    def signal_to_noise(self):
        snr = {}
        for fast_trace_type in ['low', 'high']:
            traces_zeroed = self.raw[fast_trace_type]['data'] - \
                np.mean(self.raw[fast_trace_type]['data'], axis=1)[:, None]
            snr[fast_trace_type] = np.log10(
                np.mean(traces_zeroed**2, axis=1)/np.std(traces_zeroed, axis=1))+1
        return snr
