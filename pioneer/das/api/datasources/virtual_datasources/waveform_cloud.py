from pioneer.common import platform
from pioneer.common.trace_processing import Binning, Decimate, Desaturate, Realign, RemoveStaticNoise, TraceProcessingCollection, Smooth, ZeroBaseline
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit_float_intensity
from pioneer.das.api.samples import EchoXYZIT, FastTrace

from typing import Any

import numpy as np

class WaveformCloud(VirtualDatasource):
    """Makes a point cloud from each data point of each waveforms."""

    def __init__(self, reference_sensor:str, dependencies:list, zero_baseline:bool=False, remove_negative_distances:bool=True, decimation:int=1, binning:int=1):
        """Constructor
            Args:
                reference_sensor (str): The name of the sensor (e.g. 'pixell_bfc').
                dependencies (list): A list of the datasource names. 
                    The only element should be a Trace datasource (e.g. 'pixell_bfc_ftrr')
                zero_baseline (bool): If True, the baseline of the waveforms are re-calibrated to be around zero.
                remove_negative_distances (bool): If True, the points behind the sensor are filtered out.
                decimation (int): Must be an integer larger than 1. Downsample by this factor by decimation.
                binning (int): Must be an integer larger than 1. Downsample by this factor by binning.
        """
        trr_ds_name = dependencies[0].split('_')[-1]
        super(WaveformCloud, self).__init__(f'xyzit-{trr_ds_name}', dependencies, None)
        self.reference_sensor = reference_sensor
        self.original_trace_datasource = dependencies[0]
        self.remove_negative_distances = remove_negative_distances

        trace_processing_list = []
        if zero_baseline:
            trace_processing_list.append(ZeroBaseline())
        if binning != 1:
            trace_processing_list.append(Binning(binning))
        if decimation != 1:
            trace_processing_list.append(Decimate(decimation))
        self.trace_processing = TraceProcessingCollection(trace_processing_list)       

    def get_extrema_positions(self, sensor, distances, ts, time_base_delays):

        indices = np.arange(sensor.specs['v']*sensor.specs['h'])

        xyz0 = sensor.get_corrected_cloud(ts, sensor.cache(sensor.specs), 'point_cloud', indices, time_base_delays)
        x0, y0, z0 = xyz0[:,0], xyz0[:,1], xyz0[:,2]

        xyz1 = sensor.get_corrected_cloud(ts, sensor.cache(sensor.specs), 'point_cloud', indices, distances)
        x1, y1, z1 = xyz1[:,0], xyz1[:,1], xyz1[:,2]

        return x0, y0, z0, x1, y1, z1

    def __getitem__(self, key:Any):

        if isinstance(key, slice):
            return self[platform.slice_to_range(key, len(self))]
        if isinstance(key, range):
            return [self[index] for index in key]

        trace_sample = self.datasources[self.original_trace_datasource][key]
        timestamp = trace_sample.timestamp

        if isinstance(trace_sample, FastTrace):
            raw = trace_sample.processed(self.trace_processing)['high']
        else:
            raw = trace_sample.processed(self.trace_processing)

        traces = raw['data']

        x0, y0, z0, x1, y1, z1 = self.get_extrema_positions(
            sensor=self.datasources[self.original_trace_datasource].sensor, 
            distances=trace_sample.max_range, 
            ts=timestamp, 
            time_base_delays=raw['time_base_delays']
        )

        grid = np.indices(traces.shape)
        positions = grid[1].flatten()
        indices = grid[0].flatten()
        amplitudes = traces.flatten()
        timestamps = raw['timestamps'][indices]

        x = x0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + x1[indices]*positions/traces.shape[-1]
        y = y0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + y1[indices]*positions/traces.shape[-1]
        z = z0[indices]*(traces.shape[-1]-positions)/traces.shape[-1] + z1[indices]*positions/traces.shape[-1]

        distances = (x**2 + y**2 + z**2)**0.5

        if self.remove_negative_distances:
            keep = np.where(positions > -raw['time_base_delays'][indices]/raw['distance_scaling'])[0]
        else:
            keep = slice(None)

        virtual_raw = np.empty(indices[keep].size, dtype=datasource_xyzit_float_intensity())
        virtual_raw['x'] = x[keep]
        virtual_raw['y'] = y[keep]
        virtual_raw['z'] = z[keep]
        virtual_raw['i'] = amplitudes[keep]
        virtual_raw['t'] = timestamps[keep]

        return EchoXYZIT(key, self, virtual_raw, timestamp)