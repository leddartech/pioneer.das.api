from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit_float_intensity
from pioneer.das.api.samples import XYZIT

from typing import Any

import copy
import numpy as np

class LCAx_XYZIT_traces_projection(VirtualDatasource):
    """ 
        LCAx traces 3d projection.

        il faut les time base delay, ou alors prendre un chiffre theorique du timming entre le debut de l'adc et le tir laser

    """

    def __init__(self, ds_type:str, trr_ds_name:str, sensor=None, remove_offset=True):
        """ Constructor
        Args:
            trr_ds_name:  LCAx sensor traces datasource name (e.g. 'eagle_tfc_trr')
            sensor: the sensor of the original data
        """

        self.trr_sensor_type, trr_pos, _ = parse_datasource_name(trr_ds_name)
        ech_ds_name = f"{self.trr_sensor_type}_{trr_pos}_ech" # needed for cache get_corrected_projection_data
        super(LCAx_XYZIT_traces_projection, self).__init__(ds_type, [trr_ds_name, ech_ds_name], None)

        self.trr_ds_name = trr_ds_name
        self.ech_ds_name = ech_ds_name
        self.sensor = sensor
        self.trace_length = None
        self.remove_offset = remove_offset
        
    def config_after_datasource_definition(self):
        self.trace_length = self.datasources[self.trr_ds_name][0].raw['data'].shape[1]
        self.nb_traces = self.datasources[self.trr_ds_name][0].raw['data'].shape[0]

        if self.trr_sensor_type=='eagle':
            # delay due to match filter, experimentaly determined (diff echo, and echo from trace on a full dataset)
            # would be better to callibrate time base delay from raw trace echoes to get it
            self.self_match_filter_offset = 4.391 # 6.851108587500001 -3
        elif self.trr_sensor_type=='lca2':
            self.self_match_filter_offset = 0.0 # no idea for LCA2
        else:
            raise ValueError(f"{self.trr_sensor_type} not implemented.")
        
        self.distances_vector = np.arange(self.trace_length, dtype='f8')
        self.distances_vector = self.distances_vector * self.sensor.distance_scaling + self.self_match_filter_offset

    def get_at_timestamp(self, timestamp:int):
            """override"""
            # since lcax use nearest_interpolator, we just try to find the index closest to timestamp
            float_index = self.datasources[self.trr_ds_name].to_float_index(timestamp)
            return self[int(np.floor(float_index))]

    def __getitem__(self, key:Any):
        """override
        Args: 
            key:            They key that this datasource was indexed with 
                            (e.g. a simgle index such as 42, or a slice such as 0\:10)
        """
        lcax_i = self.datasources[self.trr_ds_name][key]

        if self.trace_length is None:
            self.config_after_datasource_definition()
        
        def fill_array2(nb_traces, nb_samples_per_trace, distance_samples, time_base_delay, directions, traces, timestamps, remove_offset):
            distances = np.tile(distance_samples, nb_traces)
            ts_tile = np.repeat(timestamps, distance_samples.shape[0])
            time_base_delays = np.repeat(time_base_delay, nb_samples_per_trace)
            distances_real = distances + time_base_delays
            directions_tiled = np.repeat(directions, distance_samples.shape[0], axis=0)
            pts = directions_tiled * np.expand_dims(distances_real, axis=1)
            if remove_offset:
                traces = traces - np.mean(traces[:, :12], axis=1).reshape(nb_traces, 1)
            pts = np.hstack([pts, traces[:, :(nb_samples_per_trace)].flatten().reshape(
                nb_traces * (nb_samples_per_trace), 1)])
            to_delete = np.arange(nb_samples_per_trace-1,
                                nb_samples_per_trace*nb_traces, nb_samples_per_trace)
            
            pts = np.delete(pts, to_delete, axis=0)
            ts_tile = np.delete(ts_tile, to_delete)
            keep = np.delete(distances_real, to_delete) > 0
            return pts[keep, :], ts_tile[keep]
        
        def convert_trace_to_point(lcax_sample):
            # le simulateur genere des raw qui ne contiennent pas un timestamps par trace
            if 'timestamps' in lcax_sample.raw:
                ts = lcax_sample.raw['timestamps']
            else:
                # on duplique celui de la frame
                t = lcax_sample.raw['timestamp']
                ts = t * np.ones(lcax_sample.raw['data'].shape[1]).astype('u8')

            directions = self.sensor.get_corrected_projection_data(lcax_sample.timestamp, self.sensor.cache(self.datasources[self.ech_ds_name][0].specs))
            pts, ts = fill_array2(self.nb_traces ,self.trace_length, self.distances_vector, self.sensor.time_base_delays, directions,lcax_sample.raw['data'], np.copy(ts), self.remove_offset)
            # R_LCA = np.array([[0, 0, 1],
			# 			[-1, 0, 0],
			# 			[0, -1, 0]],
			# 			dtype=np.float)
            # pts[:,:3] = (R_LCA @ pts[:,:3].T).T
            # il faut metre le tout dans un struct array donc ajouter un colonne de timestamp
            start = lcax_sample.raw['timestamp']

            frame_raw = np.empty(pts.shape[0], dtype=datasource_xyzit_float_intensity())
            frame_raw['x'] = pts[:, 0]
            frame_raw['y'] = pts[:, 1]
            frame_raw['z'] = pts[:, 2]
            frame_raw['i'] = pts[:, 3] # - pts[:, 3].min()
            frame_raw['t'] = ts
            
            sample = XYZIT(lcax_sample.index, self, frame_raw)

            return  sample


        if isinstance(lcax_i, list):
            l = []
            for i in range(len(lcax_i)):
                l.append(convert_trace_to_point(lcax_i[i]))
        else:
            return convert_trace_to_point(lcax_i)
