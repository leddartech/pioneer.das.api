from pioneer.das.api.datasources.virtual_datasources.virtual_datasource import VirtualDatasource
from pioneer.das.api.datasources.virtual_datasources.bounding_box_metadata import BoundingBoxMetadata
from pioneer.das.api.datasources.virtual_datasources.cylindrical_projection import FlirCylindricalProjection
from pioneer.das.api.datasources.virtual_datasources.echoes_from_traces import Echoes_from_Traces
from pioneer.das.api.datasources.virtual_datasources.echoes_with_pulses import EchoesWithPulses
from pioneer.das.api.datasources.virtual_datasources.hdr_waveform import HDRWaveform
from pioneer.das.api.datasources.virtual_datasources.point_cloud_fusion import PointCloudFusion
from pioneer.das.api.datasources.virtual_datasources.rgb_cloud import RGBCloud
from pioneer.das.api.datasources.virtual_datasources.waveform_cloud import WaveformCloud
from pioneer.das.api.datasources.virtual_datasources.voxel_map import VoxelMap

VIRTUAL_DATASOURCE_FACTORY = {
    'bounding_box_metadata': BoundingBoxMetadata,
    'cylindrical_projection': FlirCylindricalProjection,
    'echoes_from_traces': Echoes_from_Traces,
    'echoes_with_pulses': EchoesWithPulses,
    'hdr_waveform': HDRWaveform,
    'point_cloud_fusion': PointCloudFusion, 
    'rgb_cloud': RGBCloud,
    'waveform_cloud': WaveformCloud,
    'voxel_map': VoxelMap,
}