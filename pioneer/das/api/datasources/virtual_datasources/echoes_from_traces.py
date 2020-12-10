from pioneer.common import clouds, peak_detector
from pioneer.common.platform import parse_datasource_name
from pioneer.common.trace_processing import Desaturate, RemoveStaticNoise, Smooth, TraceProcessingCollection, ZeroBaseline
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Echo, FastTrace

from typing import Any

import copy
import numpy as np

class Echoes_from_Traces(VirtualDatasource):
    """Non official implementation of the peak detector. Also measures the widths and the skews for each echo."""

    def __init__(self, reference_sensor:str, dependencies:list, nb_detections_max:int=3, min_amplitude:float=0):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    The only element should be a Trace datasource (e.g. 'pixell_bfc_ftrr') 
                nb_detections_max (int): The maximum number of echoes per waveform.
                min_amplitude (float): Amplitude threshold under which the echoes are filtered out.
        """
        trr_ds_name = dependencies[0].split('_')[-1].split('-')[-1]
        super(Echoes_from_Traces, self).__init__(f'ech-{trr_ds_name}', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_trace_datasource = dependencies[0]
        self.nb_detections_max = nb_detections_max
        self.peak_detector = peak_detector.PeakDetector(nb_detections_max=self.nb_detections_max, min_amplitude=min_amplitude)
        self.amplitude_scaling = 1

        self.trace_processing = None

    def _set_trace_processing(self):
        sensor = self.datasources[self.dependencies[0]].sensor
        self.trace_processing = TraceProcessingCollection([
            Desaturate(sensor.saturation_calibration),
            RemoveStaticNoise(sensor.static_noise),
            ZeroBaseline(),
            Smooth(),
        ])

    def initialize_local_cache(self, data):
        self.local_cache = copy.deepcopy(data)

    def get_echoes(self, processed_traces):
        try:
            data = processed_traces
            self.local_cache['data'][...] = data['data']
            self.local_cache['time_base_delays'][...] = data['time_base_delays']
            self.local_cache['distance_scaling'][...] = data['distance_scaling']
        except:
            self.initialize_local_cache(processed_traces)

        traces = {'data': self.local_cache['data'], 
                  'time_base_delays': self.local_cache['time_base_delays'],
                  'distance_scaling': self.local_cache['distance_scaling']}

        echoes = self.peak_detector(traces)
        echoes['amplitudes'] *= self.amplitude_scaling

        additionnal_fields = {}
        for key in echoes:
            if key not in ['indices','distances','amplitudes','timestamps','flags']:
                additionnal_fields[key] = [echoes[key], 'f4']
        return echoes, additionnal_fields

    def get_echoes_from_fast_traces(self, processed_fast_traces):
        # TODO: improve merging by replacing the saturated lines and columns
        sensor = self.datasources[self.dependencies[0]].sensor
        echoes_high, additionnal_fields_high = self.get_echoes(processed_fast_traces['high'])
        echoes_low, additionnal_fields_low = self.get_echoes(processed_fast_traces['low'])
        echoes = {}
        for field in ['indices','distances','amplitudes']:
            echoes[field] = np.hstack([echoes_high[field], echoes_low[field]])
        additionnal_fields = {}
        for field in additionnal_fields_high:
            if field not in ['indices','distances','amplitudes']:
                additionnal_fields[field] = [np.hstack([additionnal_fields_high[field][0], additionnal_fields_low[field][0]])
                                            , additionnal_fields_high[field][1]]
        return echoes, additionnal_fields

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.original_trace_datasource].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    def __getitem__(self, key:Any):

        if self.trace_processing is None:
            self._set_trace_processing()

        #Load data in local cache to prevent modifying the original data
        trace_sample = self.datasources[self.original_trace_datasource][key]
        timestamp = trace_sample.timestamp
        specs = trace_sample.specs
        if isinstance(trace_sample, FastTrace):
            get_echoes = self.get_echoes_from_fast_traces
        else:
            get_echoes = self.get_echoes

        processed_traces = trace_sample.processed(self.trace_processing)
        echoes, additionnal_fields = get_echoes(processed_traces)

        raw = clouds.to_echo_package(
            indices = np.array(echoes['indices'], 'u4'), 
            distances = np.array(echoes['distances'], 'f4'), 
            amplitudes = np.array(echoes['amplitudes'], 'f4'),
            additionnal_fields = additionnal_fields,
            timestamps = np.full(echoes['indices'].shape, timestamp), 
            flags = None, 
            timestamp = timestamp,
            specs = {"v" : specs['v'], "h" : specs['h'], "v_fov" : specs['v_fov'], "h_fov" : specs['h_fov']},
            distance_scale = 1.0, 
            amplitude_scale = 1.0, 
            led_power = 1.0, 
            eof_timestamp = None)
        
        return Echo(trace_sample.index, self, virtual_raw = raw, virtual_ts = timestamp)

