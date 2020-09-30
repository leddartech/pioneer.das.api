from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.samples.trace import Trace

from typing import Callable

import numpy as np
import copy


class FastTrace(Trace):
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(FastTrace, self).__init__(index, datasource, virtual_raw, virtual_ts)
    
    @property
    def raw(self):
        if self._raw is None:
            raw = super(Trace, self).raw

            try:
                self._raw = {
                    self.datasource.sensor.FastTraceType.LowRange: raw['low'],
                    self.datasource.sensor.FastTraceType.MidRange: raw['high'],
                }
            except:
                LoggingManager.instance().warning('Fast Traces compatibility mode enabled')
                
                self._raw = {
                    self.datasource.sensor.FastTraceType.LowRange: raw,
                    self.datasource.sensor.FastTraceType.MidRange: raw,
                }
            for fast_trace_type in self.datasource.sensor.FastTraceType:
                self._raw[fast_trace_type]['distance_scaling'] = self.datasource.sensor.distance_scaling
                self._raw[fast_trace_type]['time_base_delays'] = self.datasource.sensor.time_base_delays[fast_trace_type]
                self._raw[fast_trace_type]['trace_smoothing_kernel'] = self.datasource.sensor.get_fast_trace_smoothing_kernel(fast_trace_type)
                
        return self._raw

    @property
    def raw_array(self):
        specs = self.specs
        rawLow = self.raw[self.datasource.sensor.FastTraceType.LowRange]['data']
        rawMid = self.raw[self.datasource.sensor.FastTraceType.MidRange]['data']
        array = np.zeros((2, specs['v']*specs['h'], rawMid.shape[-1]))
        array[0] = rawMid
        array[1,:,:rawLow.shape[-1]] = rawLow
        return array.reshape((2, specs['v'], specs['h'], array.shape[-1]))

    def processed(self, trace_processing:Callable):
        processed_traces = {}
        raw_copy = copy.deepcopy(self.raw)
        for fast_trace_type in self.datasource.sensor.FastTraceType:
            processed_traces[fast_trace_type] = trace_processing(raw_copy[fast_trace_type])
        return processed_traces
