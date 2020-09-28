from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datasources.virtual_datasources.cylindrical_projection import FlirCylindricalProjection
from pioneer.das.api.datasources.virtual_datasources.echoes_from_traces import Echoes_from_Traces
from pioneer.das.api.datasources.virtual_datasources.lcax_XYZIT_traces_projection import LCAx_XYZIT_traces_projection
from pioneer.das.api.datasources.virtual_datasources.voxel_map import VoxelMap

VIRTUAL_DATASOURCE_FACTORY = {
    'cylindrical_projection': FlirCylindricalProjection,
    'echoes_from_traces': Echoes_from_Traces,
    # 'lcax_XYZIT_traces_projection': LCAx_XYZIT_traces_projection,
    'voxel_map': VoxelMap,
}