from pioneer.das.api.samples.image import Image


class ImageCylinder(Image):
    
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(ImageCylinder, self).__init__(index, datasource, virtual_raw, virtual_ts)
        self.cylindrical_projection = datasource.cylindrical_projection

    def project_pts(self, pts, mask_fov=False, output_mask=False, undistorted=False):
        ''' project 3D in the 2D cylindrical referiencial

            Args:
                pts_3D: 3D point in the center camera referential (3xN)
                mask_fov (optionnal): removes points outside the fov
                output_mask (optionnal): if True, returns the mask applied to the points
                undistorted (=False): Does nothing in the case of ImageCylinder, as it is always undistorted
            Return:
                2xN: 2D points in cylindrical image referential
                mask (optionnal): returned if output_mask is True
        '''
        return self.cylindrical_projection.project_pts(pts.T, mask_fov, output_mask)

    def undistort_image(self):
        return self.raw_image()

