from pioneer.common.platform import parse_datasource_name
from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datatypes import datasource_xyzit
from pioneer.das.api.samples import XYZIT

from typing import Any

import numpy as np

POINTPILLARS_BOX_SOURCE = 'deepen'

class PointPillars_XYZIT(VirtualDatasource):

    def __init__(self, ds_type, original_ds_name, box_ds_name, sensor=None):
        super(PointPillars_XYZIT, self).__init__(ds_type, [original_ds_name, box_ds_name], None)
        self.original_ds_name = original_ds_name
        self.box_ds_name = box_ds_name

    @staticmethod
    def add_all_combinations_to_platform(pf:'Platform') -> list:
        try:
            dss = pf.expand_wildcards(["eagle_*_ech","ouster64_*_xyzit"])
            virtual_ds_list = []
            for ds_full_name in dss:
                name, pos, ds_type = parse_datasource_name(ds_full_name)
                sensor = pf[f"{name}_{pos}"]
                box_ds_name = f'{name}_{pos}_box3d-{POINTPILLARS_BOX_SOURCE}'
                virtual_ds_type = f"xyzit-pp"
                try:
                    vds = PointPillars_XYZIT(ds_type=virtual_ds_type, original_ds_name=ds_full_name, box_ds_name=box_ds_name, sensor=sensor)
                    sensor.add_virtual_datasource(vds)
                    virtual_ds_list.append(f"{name}_{pos}_{virtual_ds_type}")
                except Exception as e:
                    print(e)
                    print(f"vitual datasource {name}_{pos}_{virtual_ds_type} was not added")
            return virtual_ds_list
        except Exception as e:
            print(e)
            print("Issue during try to add virtual datasources PointPillars_XYZIT.")

    def __getitem__(self, key:Any):
        
        pc_sample = self.datasources[self.original_ds_name][key]
        pcloud = pc_sample.point_cloud()
        box_sample = self.datasources[self.box_ds_name][key]
        box3d = box_sample.raw['data']

        ts = pc_sample.timestamp
        raw = np.empty((pcloud.shape[0]),dtype=datasource_xyzit())
        pcloud = np.matmul(pcloud,self.sensor.orientation)
        raw['x'] = pcloud[:,0]
        raw['y'] = pcloud[:,1]
        raw['z'] = pcloud[:,2]
        raw['i'] = pc_sample.amplitudes
        raw['t'] = pc_sample.timestamps

        return XYZIT(index=key, datasource=self, virtual_raw=raw, virtual_ts=ts)
