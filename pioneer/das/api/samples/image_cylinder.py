from pioneer.das.api.samples.image import Image


class ImageCylinder(Image):
    
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(ImageCylinder, self).__init__(index, datasource, virtual_raw, virtual_ts)

        self.cylindrical_projection = datasource.cylindrical_projection

    def project_pts(self, pts, output_mask=None, undistorted=False):
        ''' project 3D in the 2D cylindrical referiencial

            Args:
                pts_3D: 3D point in the center camera referential (3xN)
            Return:
                2xN: 2D points in cylindrical image referential
                keep: boolean array of point in the cylinder FOV
        '''
        return self.cylindrical_projection.project_pts(pts.T, output_mask)

    def undistort_image(self):
        return self.raw_image()

