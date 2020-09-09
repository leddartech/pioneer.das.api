from pioneer.common import clouds
from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import seg2d
from pioneer.das.api.samples import Echo, Seg2d

from typing import Any

import cv2
import numpy as np
import random

class Seg2d_from_Seg3d(VirtualDatasource):

    def __init__(self, ds_type, seg3d_ds_name, ech_ds_name, sensor=None, simplify=False, add_noise=0):
        super(Seg2d_from_Seg3d, self).__init__(ds_type, [seg3d_ds_name, ech_ds_name], None)
        self.ds_type = ds_type
        self.seg3d_ds_name = seg3d_ds_name
        self.ech_ds_name = ech_ds_name
        self.sensor = sensor
        self.simplify = simplify
        self.add_noise = add_noise

    @staticmethod
    def add_all_combinations_to_platform(pf:'Platform', simplify:bool=False, add_noise:float=0) -> list:
        try:
            seg3d_dss = pf.expand_wildcards(["*_seg3d*"]) # look for all leddar with traces
            virtual_ds_list = []

            for seg3d_ds_name_full_name in seg3d_dss:
                
                seg3d_ds_name, pos, ds_type = parse_datasource_name(seg3d_ds_name_full_name)
                sensor = pf[f"{seg3d_ds_name}_{pos}"]
                virtual_ds_type = f"seg2d-{ds_type}"
                ech_ds_name = f'{seg3d_ds_name}_{pos}_ech'

                try:
                    vds = Seg2d_from_Seg3d(
                            ds_type = virtual_ds_type,
                            seg3d_ds_name = seg3d_ds_name_full_name,
                            ech_ds_name = ech_ds_name,
                            sensor = sensor,
                            simplify = simplify,
                            add_noise = add_noise,
                    )
                    sensor.add_virtual_datasource(vds)
                    virtual_ds_list.append(f"{seg3d_ds_name}_{pos}_{virtual_ds_type}")
                except Exception as e:
                    print(e)
                    print(f"vitual datasource {seg3d_ds_name}_{pos}_{virtual_ds_type} was not added")
                
            return virtual_ds_list
        except Exception as e:
            print(e)
            print("Issue during try to add virtual datasources Seg2d_from_Seg3d.")

    def get_at_timestamp(self, timestamp):
        sample = self.datasources[self.seg3d_ds_name].get_at_timestamp(timestamp)
        return self[int(np.round(sample.index))]

    def __getitem__(self, key:Any):

        seg3d_sample = self.datasources[self.seg3d_ds_name][key]
        ts = seg3d_sample.timestamp
        ech_sample = self.datasources[self.ech_ds_name].get_at_timestamp(ts)
        echoes = ech_sample.raw['data']

        classes = seg3d_sample.raw['data']['classes']

        raw_seg2d = np.empty(0, dtype=seg2d())

        for category in np.unique(classes):

            specs = ech_sample.specs
            raw_echoes_with_category = clouds.to_echo_package(
                indices = echoes['indices'],
                distances = echoes['distances'],
                amplitudes = echoes['amplitudes'],
                additionnal_fields = {'category_mask':[np.array(classes == category, dtype=int), int]},
                timestamp = ts,
                specs = {"v" : specs['v'], "h" : specs['h'], "v_fov" : specs['v_fov'], "h_fov" : specs['h_fov']},
                )
            ech_with_category_sample = Echo(ech_sample.index, self, virtual_raw = raw_echoes_with_category, virtual_ts = ts)

            seg_img = ech_with_category_sample.other_field_img('category_mask').astype(np.uint8)

            # Find polygon and fill holes
            contours, _ = cv2.findContours(seg_img, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

            # Back to mask
            seg_img *= 0
            for contour in contours:

                # Remove points from polygon
                if self.simplify:
                    step = 4 if contour.shape[0] > 8 else 1
                    contour = np.vstack([contour[:1],contour[1:-1][::step],contour[-1:]])

                # Add noise
                if self.add_noise > 0:
                    std = self.add_noise
                    contour[...,0] += int(np.random.normal(0,std))
                    contour[...,1] += int(np.random.normal(0,std))
                    contour[...,0] = np.clip(contour[...,0], 0, seg_img.shape[1])
                    contour[...,1] = np.clip(contour[...,1], 0, seg_img.shape[0])

                cv2.drawContours(seg_img,[contour],0,1,-1)

            raw_seg2d = np.append(raw_seg2d, np.array((seg_img, category), dtype=seg2d()))

        raw = {'data':raw_seg2d}

        return Seg2d(seg3d_sample.index, self, virtual_raw = raw, virtual_ts = ts)
