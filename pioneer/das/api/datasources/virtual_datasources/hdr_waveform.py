from pioneer.common import platform
from pioneer.common.trace_processing import Realign, TraceProcessingCollection, ZeroBaseline
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Trace

from typing import Any

import copy
import numpy as np

class HDRWaveform(VirtualDatasource):
    """High Dynamic Range waveforms are created by combining both sets of waveforms from FastTraces."""

    def __init__(self, reference_sensor:str, dependencies:list, gain:float=1):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    The only element should be a FastTrace datasource (e.g. 'pixell_bfc_ftrr')
                gain (float): factor by which the low intensity channel is multiplied before being added to the high intensity channel.
        """
        trr_ds_name = dependencies[0].split('_')[-1].split('-')[-1]
        super(HDRWaveform, self).__init__(f'trr-hdr', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_trace_datasource = dependencies[0]
        self.gain = gain

        self.trace_processing = TraceProcessingCollection([ZeroBaseline(), Realign()])
        self.target_time_base_delay = None

    def __getitem__(self, key:Any):

        if isinstance(key, slice):
            return self[platform.slice_to_range(key, len(self))]
        if isinstance(key, range):
            return [self[index] for index in key]

        trace_sample = self.datasources[self.original_trace_datasource][key]
        timestamp = trace_sample.timestamp

        if self.target_time_base_delay is None:
            self.target_time_base_delay = max([
                np.max(trace_sample.raw['high']['time_base_delays']),
                np.max(trace_sample.raw['low']['time_base_delays']),
            ])
            self.trace_processing = TraceProcessingCollection([ZeroBaseline(), Realign(self.target_time_base_delay)])

        processed_traces = trace_sample.processed(self.trace_processing)

        raw = copy.deepcopy(processed_traces['high'])

        low_traces = processed_traces['low']['data']
        raw['data'][:,:low_traces.shape[-1]] += self.gain*low_traces

        return Trace(key, self, raw, timestamp)