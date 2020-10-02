from pioneer.common.trace_processing import TraceProcessingCollection, ZeroBaseline, Realign
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Trace

from typing import Any

import numpy as np
import copy



class HDRWaveform(VirtualDatasource):

    def __init__(self, reference_sensor, dependencies):
        trr_ds_name = dependencies[0].split('_')[-1].split('-')[-1]
        super(HDRWaveform, self).__init__(f'trr-hdr', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_trace_datasource = dependencies[0]

        self.trace_processing = TraceProcessingCollection([ZeroBaseline(), Realign()])
        self.target_time_base_delay = None

    def __getitem__(self, key:Any):

        trace_sample = self.datasources[self.original_trace_datasource][key]
        timestamp = trace_sample.timestamp

        if self.target_time_base_delay is None:
            self.target_time_base_delay = max([
                trace_sample.raw['high']['time_base_delays'].max(),
                trace_sample.raw['low']['time_base_delays'].max(),
            ])
            self.trace_processing = TraceProcessingCollection([ZeroBaseline(), Realign(self.target_time_base_delay)])

        processed_traces = trace_sample.processed(self.trace_processing)

        raw = copy.deepcopy(processed_traces['high'])

        low_traces = processed_traces['low']['data']
        raw['data'][:,:low_traces.shape[-1]] += low_traces

        return Trace(key, self, raw, timestamp)