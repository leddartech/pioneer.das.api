from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.interpolators import floor_interpolator
from pioneer.das.api.samples import XYZIT

from typing import Any

import numpy as np

class LCAx_XYZIT_Synchronized(VirtualDatasource):
    """Virtual datasource helper for LCA - xyzit
    
        Example timelines a trigged LCA and a rolling xyzit (b: begin, e: end, *: eagle sample, .: xyzit sample) 
        
        - In clip_to_ts = 'default' mode:
                             b          e b           e
        ech                : |**********| |***********|
                            b   b   b   b   b   b   b   b
        xyzit             : |...|...|...|...|...|...|...|
                             b          e b           e
        xyzit-eagle-tfc   :  |..........| |...........|

        - In clip_to_ts = 'aligned' mode:
                             b          e b           e
        ech                : |**********| |***********|
                            b   b   b   b   b   b   b   b
        xyzit             : |...|...|...|...|...|...|...|
                             b   e        b   e
        xyzit-eagle-tfc   :  |...|        |...|

        - In clip_to_ts = 'floor' mode:
                             b          e b           e
        ech                : |**********| |***********|
                            b   b   b   b   b   b   b   b
        xyzit             : |...|...|...|...|...|...|...|
                            b   e       b   e
        xyzit-eagle-tfc   : |...|       |...|
    
    """
    def __init__(self, ds_type:str, lcax_ds_name:str, xyzit_ds_name:str, clip_to_fov:bool = False, clip_to_ts:str = 'default'):
        """ Constructor

        Args:
            lcax_ds_name:  LCAx sensor datasource name (e.g. 'eagle_tfc_ech')
            xyzit_ds_name: XYZIT sensor datasource name (e.g. 'ouster64_tfc_xyzit')
        """
        super(LCAx_XYZIT_Synchronized, self).__init__(ds_type, [lcax_ds_name, xyzit_ds_name], None)

        self.lcax_ds_name = lcax_ds_name
        self.xyzit_ds_name = xyzit_ds_name
        self.clip_to_fov = clip_to_fov
        self.clip_to_ts = clip_to_ts

    @staticmethod
    def add_all_combinations_to_platform(pf:'Platform', add_clip:bool=True) -> list:
        ech_dss = pf.expand_wildcards(["*_ech"])
        xyzit_dss = pf.expand_wildcards(["*_xyzit"])
        virtual_ds_list = []
        for xyzit_ds_name in xyzit_dss:
            sensor_type, pos, ds_type = parse_datasource_name(xyzit_ds_name)
            sensor = pf[f"{sensor_type}_{pos}"] #it is curcial to make sure that the sensor to which 
            # the datasource is added is the one in which referential the (3D) data is represented

            for ech_ds_name in ech_dss:
                ech_sensor_type, ech_pos, _ = parse_datasource_name(ech_ds_name)
                virtual_ds_type = f"xyzit-{ech_sensor_type}-{ech_pos}"
                sensor.add_virtual_datasource(LCAx_XYZIT_Synchronized(virtual_ds_type, ech_ds_name, xyzit_ds_name))
                virtual_ds_list.append(f"{sensor_type}_{pos}_{virtual_ds_type}")
                if add_clip:
                    virtual_ds_type = f"xyzit-{ech_sensor_type}-{ech_pos}-clipped"
                    clip_to_ts = 'aligned' if ('eagle' in ech_sensor_type) else 'floor'
                    sensor.add_virtual_datasource(LCAx_XYZIT_Synchronized(virtual_ds_type, ech_ds_name, xyzit_ds_name, True, clip_to_ts))
                    virtual_ds_list.append(f"{sensor_type}_{pos}_{virtual_ds_type}")

        return virtual_ds_list

    def get_at_timestamp(self, timestamp:int):
        """override"""
        # since lcax use nearest_interpolator, we just try to find the index closest to timestamp
        float_index = self.datasources[self.lcax_ds_name].to_float_index(timestamp)
        return self[int(np.floor(float_index))]

    def __getitem__(self, key:Any):
        """override
        Args: 
            key:            They key that this datasource was indexed with 
                            (e.g. a simgle index such as 42, or a slice such as 0\:10)
        """

        lcax_i = self.datasources[self.lcax_ds_name][key]

        xyzit = self.datasources[self.xyzit_ds_name]
        
        def get_overlapping_data(lcax_sample):
            start, end = [lcax_sample.raw[l]  for l in ['timestamp', 'eof_timestamp']]

            xyzit_from = xyzit.get_at_timestamp(start, interpolator=floor_interpolator) 
            xyzit_to = xyzit.get_at_timestamp(end, interpolator=floor_interpolator)

            xyzit_from.index = int(np.floor(xyzit_from.index))
            xyzit_to.index = int(np.floor(xyzit_to.index))
            n_samples = xyzit_to.index - xyzit_from.index + 1

            parts = [np.empty((0,), dtype = xyzit_to.raw.dtype) ]

            if self.clip_to_ts == 'aligned':

                if n_samples >= 1:
                    parts.append(xyzit_from.raw[(xyzit_from.raw['t'] >= start) & (xyzit_from.raw['t'] <= end)])
                if n_samples >= 2:
                    dt = xyzit_to.timestamp - xyzit_from.timestamp
                    xyzit_next = xyzit[xyzit_from.index+1]
                    parts.append(xyzit_next.raw[(xyzit_next.raw['t'] <= start + dt)])
            elif self.clip_to_ts == 'floor':
                parts.append(xyzit_from.raw)
            else: #'default'
                if n_samples == 1:
                    parts.append(xyzit_from.raw[(xyzit_from.raw['t'] >= start) & (xyzit_from.raw['t'] <= end)])
                elif n_samples >= 2:
                    parts.append(xyzit_from.raw[(xyzit_from.raw['t'] >= start)])
                    if n_samples > 2:
                        xyzit_samples = xyzit[xyzit_from.index+1:xyzit_to.index]
                        for xyzit_sample in xyzit_samples:
                            parts.append(xyzit_sample.raw)
                    parts.append(xyzit_to.raw[xyzit_to.raw['t'] <= end])

            
            raw = np.hstack(parts)

            sample = XYZIT(lcax_sample.index, self, raw, start)

            if self.clip_to_fov:

                pts = sample.point_cloud(referential=self.lcax_ds_name, undistort=False)
                
                mask = lcax_sample.clip_to_fov_mask(pts)

                sample.virtual_raw = sample.virtual_raw[mask]

            return  sample


        if isinstance(lcax_i, list):
            l = []
            for i in range(len(lcax_i)):
                l.append(get_overlapping_data(lcax_i[i]))
        else:
            return get_overlapping_data(lcax_i)