
from pioneer.common import linalg
from pioneer.common.cylindrical_projection import CylindricalProjection, Pos
from pioneer.das.api.datasources.virtual_datasources import VirtualDatasource
from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples import ImageCylinder, Seg2dImage

from enum import Enum

import cv2
import numpy as np

class FlirCylindricalProjection(VirtualDatasource):

    def __init__(self
        , name
        , cameras
        , intrinsics_calibrations
        , distortion_coefficients
        , extrinsic_calibrations
        , datatype = 'img'
        , config={'radius':50.0, 'FOV_h':np.deg2rad(210), 'FOV_v':np.deg2rad(67.5), 'image_h':2000, 'image_v':int(2000*0.25), 'fusion_overlap_ratio': 0.25}
        ):

        cameras = list(map(lambda camera: f"{camera}_{datatype}", cameras))
        super(FlirCylindricalProjection, self).__init__(name, cameras, None, timestamps_source = cameras[Pos.CENTER.value])

        if 'seg2d' in datatype:
            config['fusion_overlap_ratio'] = 0.0
            
        self.cylindrical_projection = CylindricalProjection(intrinsics_calibrations, distortion_coefficients, extrinsic_calibrations, config)

    @staticmethod
    def add_to_platform(pf:'Platform', cameras=['flir_bfl', 'flir_bfc', 'flir_bfr'], datatype='img', remove_offset:bool=True) -> list:
        """
            Parameters
                pf: platform
                cameras: A list of 3 cameras in order [left, center, right]
            Returns a virtual datasource for cylindrical projection
        """
        def validate_cameras(cameras, pf):
            if len(cameras) != 3:
                raise Exception("You need 3 cameras to be able to use the cylindrical projection.")

            for camera in cameras:
                if camera not in pf.sensors:
                    raise Exception(f"Cannot find sensor {camera} in the platform.")
        try:
            validate_cameras(cameras, pf)
            
            intrinsics_calibrations = list(map(lambda camera: pf.intrinsics[camera]['matrix'], cameras))
            distortion_coefficients = list(map(lambda camera: pf.intrinsics[camera]['distortion'], cameras))
            extrinsic_calibrations  = list(map(lambda camera: pf.extrinsics[camera][cameras[Pos.CENTER.value]], cameras))

            sensor = pf[cameras[Pos.CENTER.value]]
            virtual_datasource_name = f"{datatype}-cyl"
            try:
                sensor.add_virtual_datasource(FlirCylindricalProjection(virtual_datasource_name, cameras, intrinsics_calibrations, distortion_coefficients, extrinsic_calibrations, datatype))
                return f"{cameras[Pos.CENTER.value]}_{virtual_datasource_name}"
            except Exception as e:
                print(e)
                print(f"vitual datasource {cameras[Pos.CENTER.value]}_{virtual_datasource_name} was not added")
                
        except Exception as e:
            print(e)
            print("Issue during try to add virtual datasources FlirCylindricalProjection.")

    def __getitem__(self, key):
        """override
        Args: 
            key:            They key that this datasource was indexed with 
                            (e.g. a simgle index such as 42, or a slice such as 0\:10)
        """

        img_c_sample = self.datasources[self.dependencies[Pos.CENTER.value]][key]
        ts = img_c_sample.timestamp

        img_c = img_c_sample.raw
        img_r = self.datasources[self.dependencies[Pos.RIGHT.value]].get_at_timestamp(ts, nearest_interpolator).raw
        img_l = self.datasources[self.dependencies[Pos.LEFT.value]].get_at_timestamp(ts, nearest_interpolator).raw

        raw = self.cylindrical_projection.stitch([img_l, img_c, img_r])

        if isinstance(img_c_sample, Seg2dImage):
            raw = raw[...,0]
            return Seg2dImage(key, self, raw, virtual_ts=ts)

        return ImageCylinder(key, self, raw, virtual_ts=ts)

