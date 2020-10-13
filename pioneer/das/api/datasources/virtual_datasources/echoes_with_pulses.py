from pioneer.common import clouds, platform
from pioneer.common.trace_processing import TraceProcessingCollection, ZeroBaseline
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Echo, FastTrace

from typing import Any

import copy
import numpy as np

class EchoesWithPulses(VirtualDatasource):
    """Echoes, with a number of data points from the waveforms corresponding to the pulses that resulted in each echo."""

    def __init__(self, reference_sensor:str, dependencies:list, pulse_sample_size:int=10, zero_baseline:bool=True):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    First element should be an Echo datasource (e.g. 'pixell_bfc_ech')
                    Second element should be a Trace datasource (e.g. 'pixell_bfc_ftrr')
                pulse_sample_size (int): The number of data points to gather before and after each echo in the waveforms.
                    For example, with pulse_sample_size=10, the pulses will be 21 points, because the highest point is
                    taken in addition to the 10 points before and the 10 points after.
                zero_baseline (bool): If True, the baseline of the waveforms are re-calibrated to be around zero.
        """
        super(EchoesWithPulses, self).__init__(f'ech-pulses', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_echoes_datasource = dependencies[0]
        self.original_trace_datasource = dependencies[1]
        self.pulse_sample_size = pulse_sample_size
        self.zero_baseline = zero_baseline

        if self.zero_baseline:
            self.trace_processing = TraceProcessingCollection([ZeroBaseline()])
        else:
            self.trace_processing = TraceProcessingCollection([])  

    def __getitem__(self, key:Any):

        if isinstance(key, slice):
            return self[platform.slice_to_range(key, len(self))]
        if isinstance(key, range):
            return [self[index] for index in key]

        echoes_sample = self.datasources[self.original_echoes_datasource][key]

        timestamp = echoes_sample.timestamp
        specs = echoes_sample.specs

        trace_sample = self.datasources[self.original_trace_datasource].get_at_timestamp(timestamp)
        if isinstance(trace_sample, FastTrace):
            raw_traces = trace_sample.processed(self.trace_processing)['high']
        else:
            raw_traces = trace_sample.processed(self.trace_processing)

        full_traces = raw_traces['data'][echoes_sample.indices]

        echoes_positions_in_traces = (echoes_sample.distances - raw_traces['time_base_delays'][echoes_sample.indices])/raw_traces['distance_scaling']
        echoes_positions_in_traces = echoes_positions_in_traces.astype(int)

        ind = np.indices(echoes_positions_in_traces.shape)
        padded_traces = np.pad(full_traces,((0,0),(self.pulse_sample_size,self.pulse_sample_size+1)))
        pulses = np.vstack([padded_traces[ind,echoes_positions_in_traces+ind_pulse+self.pulse_sample_size] for ind_pulse in np.arange(-self.pulse_sample_size, self.pulse_sample_size+1, dtype=int)]).T

        indices = np.repeat(echoes_sample.indices[...,None], pulses.shape[-1], axis=-1).flatten()
        delta = self.pulse_sample_size*raw_traces['distance_scaling']
        deltas = np.repeat(np.linspace(-delta, delta, pulses.shape[-1])[None,...], pulses.shape[0], axis=0)
        distances = (np.repeat(echoes_sample.distances[...,None], pulses.shape[-1], axis=-1) - deltas).flatten()
        amplitudes = pulses.flatten()
        timestamps = (np.repeat(echoes_sample.timestamps[...,None], pulses.shape[-1], axis=-1)).flatten()
        flags = (np.repeat(echoes_sample.flags[...,None], pulses.shape[-1], axis=-1)).flatten()

        keep = np.where((distances > 0) & (amplitudes > 0))[0]

        raw = clouds.to_echo_package(
            indices = indices[keep], 
            distances = distances[keep], 
            amplitudes = amplitudes[keep],
            timestamps = timestamps[keep], 
            flags = flags[keep], 
            timestamp = timestamp,
            specs = {"v" : specs['v'], "h" : specs['h'], "v_fov" : specs['v_fov'], "h_fov" : specs['h_fov']},
            distance_scale = 1.0, 
            amplitude_scale = 1.0, 
            led_power = 1.0, 
            eof_timestamp = None)
        
        return Echo(key, self, raw, timestamp)
