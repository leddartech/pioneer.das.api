from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.samples import Trace

from typing import Any

import numpy as np

class Traces_from_Echoes(VirtualDatasource):

    def __init__(self, ds_type, ech_ds_name, sensor=None, use_widths=True, use_skews=True, saturation=None):
        super(Traces_from_Echoes, self).__init__(ds_type, [ech_ds_name], None)
        self.ds_type = ds_type
        self.ech_ds_name = ech_ds_name
        self.sensor = sensor
        self.amplitude_scaling = 1
        self.use_widths = use_widths
        self.use_skews = use_skews
        self.saturation = saturation

        self.time_base_delays = self.sensor.time_base_delays
        if type(self.time_base_delays) in [int, float] or self.time_base_delays.shape[0] == 1:
            self.time_base_delays = np.full(self.sensor.specs['h']*self.sensor.specs['v'], self.time_base_delays)

        #FIXME: hardcoded amplitude scaling between traces and echoes
        if 'eagle' in self.sensor.name:
            self.amplitude_scaling = 20
        elif 'lca2' in self.sensor.name:
            self.amplitude_scaling = 20

        #Filter: (Gaussian-like with an undershoot)
        self.pulse_width = 0.35
        if 'lca2' in self.sensor.name:
            self.pulse_width = 0.11

    @staticmethod
    def add_all_combinations_to_platform(pf:'Platform', use_widths:bool=True, use_skews:bool=True, saturation=None) -> list:
        try:
            ech_dss = pf.expand_wildcards(["*_ech*"]) # look for all leddar with traces
            virtual_ds_list = []

            for ech_ds_name_full_name in ech_dss:
                
                ech_ds_name, ech_pos, ds_type = parse_datasource_name(ech_ds_name_full_name)
                sensor = pf[f"{ech_ds_name}_{ech_pos}"]
                virtual_ds_type = f"trr-{ds_type}"

                try:
                    vds = Traces_from_Echoes(
                            ds_type = virtual_ds_type,
                            ech_ds_name = ech_ds_name_full_name,
                            sensor = sensor,
                            use_widths = use_widths,
                            use_skews = use_skews,
                            saturation = saturation,
                    )
                    sensor.add_virtual_datasource(vds)
                    virtual_ds_list.append(f"{ech_ds_name}_{ech_pos}_{virtual_ds_type}")
                except Exception as e:
                    print(e)
                    print(f"vitual datasource {ech_ds_name}_{ech_pos}_{virtual_ds_type} was not added")
                
            return virtual_ds_list
        except Exception as e:
            print(e)
            print("Issue during try to add virtual datasources Traces_from_Echoes.")


    def compute_response_simple_gaussian(self, echoes):
        traces = np.zeros((self.sensor.specs['h']*self.sensor.specs['v'],self.sensor.oversampling*self.sensor.base_point_count), dtype=np.float)
        distances = np.linspace(start = self.time_base_delays[echoes['indices']],
                                stop = self.time_base_delays[echoes['indices']]+traces.shape[-1]*self.sensor.distance_scaling,
                                num = traces.shape[-1], axis = -1)

        if self.use_widths and 'widths' in echoes.dtype.names:
            w_left = echoes['widths'][:,None]/4
            w_right = echoes['widths'][:,None]/4
        else: 
            w_left, w_right = 3, 3

        if self.use_skews and 'skews' in echoes.dtype.names:
            w_left += echoes['skews'][:,None]
            w_right -= echoes['skews'][:,None]

        pulses_left = echoes['amplitudes'][:,None]*np.exp(-(distances-echoes['distances'][:,None])**2/(2*w_left**2))
        pulses_right = echoes['amplitudes'][:,None]*np.exp(-(distances-echoes['distances'][:,None])**2/(2*w_right**2))
        peak_indices = np.argmin(np.abs(distances-echoes['distances'][:,None]), axis=1)

        for i, (ch,peak_indice) in enumerate(zip(echoes['indices'],peak_indices)):
            traces[ch,:peak_indice] += pulses_left[i,:peak_indice]
            traces[ch,peak_indice:] += pulses_right[i,peak_indice:]

        return traces*self.amplitude_scaling


    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.ech_ds_name].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    def __getitem__(self, key:Any):

        echoes_sample = self.datasources[self.ech_ds_name][key]
        echoes = echoes_sample.data
        ts = echoes_sample.timestamp

        traces = self.compute_response_simple_gaussian(echoes)

        if self.sensor.static_noise is not None:
            traces += self.sensor.static_noise

        if self.saturation is not None:
            traces = np.clip(traces, 0, self.saturation)

        raw = {'data':traces, 'start_index':0, 'timestamp':ts, 'time_base_delays':self.sensor.time_base_delays}

        try: raw['scan_direction'] = echoes_sample.raw['scan_direction']
        except: pass
        
        return Trace(index=key, datasource=self, virtual_raw=raw, virtual_ts=ts)
