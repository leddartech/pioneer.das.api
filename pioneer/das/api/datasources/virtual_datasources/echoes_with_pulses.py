from pioneer.common import clouds, platform
from pioneer.common.trace_processing import Binning, Decimate, TraceProcessingCollection, ZeroBaseline
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Echo, FastTrace

from typing import Any

import copy
import numpy as np

class EchoesWithPulses(VirtualDatasource):
    """Echoes, with a number of data points from the waveforms corresponding to the pulses that resulted in each echo."""

    def __init__(self, reference_sensor:str, dependencies:list, pulse_sample_size:int=10, zero_baseline:bool=True, decimation:int=1, binning:int=1, pulses_as_echoes:bool=True):
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
                decimation (int): Must be an integer larger than 1. Downsample the pulses by this factor by decimation.
                binning (int): Must be an integer larger than 1. Downsample the pulses by this factor by binning.
                pulses_as_echoes (bool): If True, each sample from each pulse is treated as an echo. 
                    If False, the echoes are untouched and the pulses are stored under the 'pulses' key in the raw dictionnary. 
        """
        super(EchoesWithPulses, self).__init__(f'ech-pulses', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_echoes_datasource = dependencies[0]
        self.original_trace_datasource = dependencies[1]
        self.pulse_sample_size = pulse_sample_size
        self.pulses_as_echoes = pulses_as_echoes

        trace_processing_list = []
        if zero_baseline:
            trace_processing_list.append(ZeroBaseline())
        if binning != 1:
            trace_processing_list.append(Binning(binning))
        if decimation != 1:
            trace_processing_list.append(Decimate(decimation))
        self.trace_processing = TraceProcessingCollection(trace_processing_list)  

    def __getitem__(self, key:Any):

        if isinstance(key, slice):
            return self[platform.slice_to_range(key, len(self))]
        if isinstance(key, range):
            return [self[index] for index in key]

        echoes_sample = self.datasources[self.original_echoes_datasource][key]

        timestamp = echoes_sample.timestamp
        specs = echoes_sample.specs

        pulses, distance_scaling = echoes_sample.get_pulses(self.original_trace_datasource.split('_')[-1], self.pulse_sample_size, self.trace_processing, return_distance_scaling=True)

        if not self.pulses_as_echoes:

            raw = clouds.to_echo_package(
                indices = echoes_sample.indices, 
                distances = echoes_sample.distances, 
                amplitudes = echoes_sample.amplitudes,
                timestamps = echoes_sample.timestamps, 
                flags = echoes_sample.flags, 
                timestamp = timestamp,
                specs = {"v" : specs['v'], "h" : specs['h'], "v_fov" : specs['v_fov'], "h_fov" : specs['h_fov']},
                distance_scale = 1.0, 
                amplitude_scale = 1.0, 
                led_power = 1.0, 
                eof_timestamp = None)

            raw['pulses'] = pulses
            
            return Echo(key, self, raw, timestamp)

        else:

            indices = np.repeat(echoes_sample.indices[...,None], pulses.shape[-1], axis=-1).flatten()
            delta = self.pulse_sample_size*distance_scaling
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