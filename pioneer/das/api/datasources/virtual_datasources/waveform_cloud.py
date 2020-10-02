from pioneer.common.trace_processing import TraceProcessingCollection, Desaturate, RemoveStaticNoise, ZeroBaseline, Smooth, Realign
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit_float_intensity
from pioneer.das.api.samples import EchoXYZIT

from typing import Any

import numpy as np



class WaveformCloud(VirtualDatasource):

    def __init__(self, reference_sensor, dependencies):
        trr_ds_name = dependencies[0].split('_')[-1]
        super(WaveformCloud, self).__init__(f'xyzit-{trr_ds_name}', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_trace_datasource = dependencies[0]

    def get_extrema_positions(self, sensor, distances, ts, time_base_delays):

        indices = np.arange(sensor.specs['v']*sensor.specs['h'])

        xyz0 = sensor.get_corrected_cloud(ts, sensor.cache(sensor.specs), 'point_cloud', indices, time_base_delays)
        x0, y0, z0 = xyz0[:,0], xyz0[:,1], xyz0[:,2]

        xyz1 = sensor.get_corrected_cloud(ts, sensor.cache(sensor.specs), 'point_cloud', indices, distances)
        x1, y1, z1 = xyz1[:,0], xyz1[:,1], xyz1[:,2]

        return x0, y0, z0, x1, y1, z1

    def __getitem__(self, key:Any):

        # TODO: get per channel timestamp to fix motion compensation

        trace_sample = self.datasources[self.original_trace_datasource][key]
        timestamp = trace_sample.timestamp
        traces = trace_sample.raw['data']

        x0, y0, z0, x1, y1, z1 = self.get_extrema_positions(
            sensor=self.datasources[self.original_trace_datasource].sensor, 
            distances=trace_sample.max_range, 
            ts=timestamp, 
            time_base_delays=trace_sample.raw['time_base_delays']
        )

        grid = np.indices(traces.shape)
        positions = grid[1].flatten()
        indices = grid[0].flatten()
        amplitudes = traces.flatten()

        x = x0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + x1[indices]*positions/traces.shape[-1]
        y = y0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + y1[indices]*positions/traces.shape[-1]
        z = z0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + z1[indices]*positions/traces.shape[-1]

        distances = (x**2 + y**2 + z**2)**0.5

        keep = np.where(positions > -trace_sample.raw['time_base_delays'][indices]/trace_sample.raw['distance_scaling'])[0]

        raw = np.empty(indices[keep].size, dtype=datasource_xyzit_float_intensity())
        raw['x'] = x[keep]
        raw['y'] = y[keep]
        raw['z'] = z[keep]
        raw['i'] = amplitudes[keep]
        raw['t'] = timestamp

        return EchoXYZIT(key, self, raw, timestamp)