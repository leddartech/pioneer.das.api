from pioneer.common import platform as platform_utils
from pioneer.common.cylindrical_projection import CylindricalProjection, Pos
from pioneer.das.api.datasources.virtual_datasources import VirtualDatasource
from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples import ImageCylinder

import numpy as np

class FlirCylindricalProjection(VirtualDatasource):
    """Merges the images of three cameras in a single one."""

    def __init__(self, reference_sensor:str, dependencies:list, 
                    radius=50.0, fov_h=np.deg2rad(210), fov_v=np.deg2rad(67.5), 
                    image_h=2000, image_v=int(2000*0.25), 
                    fusion_overlap_ratio=0.25):
        """Constructor
            Args:
                reference_sensor (str): The name of the central camera (e.g. 'flir_bfc').
                dependencies (list): A list of the image datasource names from left to right. 
                    (e.g. ['flir_bfl_img','flir_bfc_img','flir_bfr_img'])
        """
        super(FlirCylindricalProjection, self).__init__('img-cyl', dependencies, None, timestamps_source = dependencies[Pos.CENTER.value])
        self.reference_sensor = reference_sensor
        self.dependencies = dependencies
        self.config={'radius':radius, 'FOV_h':np.deg2rad(fov_h), 'FOV_v':np.deg2rad(fov_v), 'image_h':image_h, 'image_v':int(image_v), 'fusion_overlap_ratio': fusion_overlap_ratio}
        self.cylindrical_projection = None
            

    def _set_cylindrical_projection(self):
        pf = self.datasources[self.dependencies[0]].sensor.platform
        cameras = list(map(lambda camera: platform_utils.extract_sensor_id(camera), self.dependencies))
        intrinsics_calibrations = list(map(lambda camera: pf.intrinsics[camera]['matrix'], cameras))
        distortion_coefficients = list(map(lambda camera: pf.intrinsics[camera]['distortion'], cameras))
        extrinsic_calibrations  = list(map(lambda camera: pf.extrinsics[camera][cameras[Pos.CENTER.value]], cameras))
            
        self.cylindrical_projection = CylindricalProjection(intrinsics_calibrations, distortion_coefficients, extrinsic_calibrations, self.config)


    def __getitem__(self, key):

        if self.cylindrical_projection is None:
            self._set_cylindrical_projection()

        img_c_sample = self.datasources[self.dependencies[Pos.CENTER.value]][key]
        ts = img_c_sample.timestamp

        img_c = img_c_sample.raw
        img_r = self.datasources[self.dependencies[Pos.RIGHT.value]].get_at_timestamp(ts, nearest_interpolator).raw
        img_l = self.datasources[self.dependencies[Pos.LEFT.value]].get_at_timestamp(ts, nearest_interpolator).raw

        raw = self.cylindrical_projection.stitch([img_l, img_c, img_r])

        return ImageCylinder(key, self, raw, virtual_ts=ts)

