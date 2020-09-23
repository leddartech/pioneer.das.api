# Extrinsic calibration of cameras with a Leddar

This document describes the process we use to compute the extrinsic calibration
between a combination of cameras and leddars. The sensor configuration used in
this document is a 20 deg HFOV Eagle sensor with three FLIR cameras. The left
and right cameras have a FOV of about 60 deg and the center camera shares much
of the Eagle FOV with a 25 deg FOV.

To calibrate the different sensors you need to follow four basic steps:

- Record calibration datasets
- Predetect chessboards in cameras
- Compute 3D points correspondence between sensors, i.e. homologs
- Optimize the rigid transformations between different pairs of sensors

# Recording calibration datasets

Calibration datasets are recorded in a static setting with a person holding a chessboard and moving in the field of view of the different sensors. Here are a few things to keep in mind when you record calibration datasets:

- Lower the VBIAS setting of your Leddars. Most of our calibration datasets are recorded with `VBIAS==100`. This prevents the sensor from saturating too much since saturation has a strong effect on the measured distance.
- Do not start too close to the sensors. The closer you are the more saturation you will get.
- Try to stay within the Leddar field of view as much as possible. When the chessboard is only partially visible the simple reflective sphere algorithm will fail and you will get many outliers.
- Record your dataset in a scene without any reflectors, we had issues with many sorts of reflectors, snow plow landmarks for example.
- Keep your hands behind the chessboard as much as  possible. Visible hands could lead to many undetected chessboards.
- Keep the chessboard vertical as much as possible. The reflective sphere detector assumes that the chessboard is roughly perpendicular to the Leddar optical axis.


## Predetecting the chessboards in camera images

The [cam_intrinsics_calibration.py ](./cam_intrinsics_calibration.py ) script
should be used to detect the chessboards in all images of each camera. To use the
script you need to add a section to your platform.yml `ignore` section. It should
look like this:
```yaml
eagle_tfc: ...
flir_tfc: ...
flir_tfr: ...
flir_tfl: ...
ignore:
    camera_intrinsics:
        # chessboard used for camera intrinsics
        chessboard:
            nx: 4 # number of columns of corners
            ny: 3 # number of rows of corners
            dx: 0.145 # width of a chessboad square
            dy: 0.145 # height of a chessboard square

            reflective_spheres: [
                {
                    # [x, y] == [col, row]
                    corner_xy: [3, 0],
                    # offsets x, y
                    offsets: [0.03, 0.03]
                }
            ]
```
The chessboard should be asymetric, i.e. it should have an even (odd) number of rows and an odd (even) number of columns.

Once this is done, we can use the following command to compute the chessboard for all cameras:
```bash
$ python3 cam_intrinsics_calibration /nas/path/to/dataset flir_tfc_img --chessboard --slow
```
It is recommended to use the `--slow` parameter to make sure that the chessboard detection algorithm detects a maximum number of chessboards. Repeat this command by replacing `flir_tfc_img` by `flir_tfl_img` and `flir_tfr_img`. You should now have three zip files named after you cameras with `_chessboards.zip` suffix, e.g. `flir_tfc_chessboards.zip`.

## Computing 3D point correspondences between sensors

We use the detected chessboards and camera intrinsics to compute 3D points in the camera reference frames (see `cv2.solvePnP`). For the Eagle sensor we detect a reflective sphere located on the chessboard by assuming that the point with the maximum amplitude corresponds to this sphere. This hypothesis is not always verified but the ouliers are usually correctly filtered by RANSAC in the next step. Also, noise in the Leddar point cloud often makes the sphere stand out of the chessboard plane so we fit a plane in the surroundings of the point and compute the intersection of the ray with the estimated plane to correct the distance.

To compute the correspondences between points you need to write a YAML file that
looks like the [homologs.yml](./homologs.yml) file:
```yaml
# dataset to use to generate homologs
# this dataset platform.yml should contain the required intrinsics section
# for the cameras and leddars. Also, the platform.yml should contain the
# chessboard description that we want to use to calibrate. Ideally you will
# predetect the chessboards in all cameras using the
# cam_intrinsics_calibration.py script
dataset: '../'
# Do not compute the homologs for all frames but instantiate a viewer where we can
# see the results of the homologs detection. We can't use this option with
# more than two sensors. So comment out some sensors below if you want to use this option.
view: false
# Set to true if you want to see the point clouds of each of the
# sensor pairs once they are computed
view_homologs: false
# possible choices ['flir_tfl', 'flir_tfc', 'flir_tfr', 'eagle_tfc']
# list the sensors for which you want to generate the point correspondences.
sensors:
  flir_tfl:
    config: &config
      # there was an attempt to detect the chessboard corners with
      # a convolutional neural network, but it did not generalize very
      # well from a calibration dataset to another. So this parameter should
      # be true at all times.
      reflective_spheres: true
  flir_tfr:
    config: *config
  flir_tfc:
    config: *config
  eagle_tfc:
    config: *config
```
This file will be read by the `homologs.py` script to compute the 3D point correspondences between cameras and leddar sensors. When the two sensors are cameras we ignore the `reflective_spheres` parameters so that all chessboard
corners are taken into account. The script is run like this:
```bash
$ python3 homologs.py /nas/path/to/folder/homologs.yml
```
This script will create `.csv` files for all pairs of sensors in the directory of the YAML file. The files are named after the sensors used to compute the point correspondences, e.g. `flir_tfc-eagle_tfc-homologs.csv`. Each line of of the csv file has 7 columns. The first column is just an index and columns 2-4 are the coordinates of a point in the first sensor in the name of the file while colums 5-7 are the coordinates of the corresponding point in the other sensor.

Also, it will create a `loop_config.yml` to be used with the extrinsic calibration script. The file will look like this:
```yaml
loops:
  loop1: flir_tfl-flir_tfr->-flir_tfl-flir_tfr
output: loop_all.pkl
plot: null
ransac:
  distance_threshold: 0.15
  nb_iterations: 5000
transformations:
  flir_tfc-eagle_tfc: flir_tfc-eagle_tfc-homologs.csv
  flir_tfl-eagle_tfc: flir_tfl-eagle_tfc-homologs.csv
  flir_tfl-flir_tfc: flir_tfl-flir_tfc-homologs.csv
  flir_tfl-flir_tfr: flir_tfl-flir_tfr-homologs.csv
  flir_tfr-eagle_tfc: flir_tfr-eagle_tfc-homologs.csv
  flir_tfr-flir_tfc: flir_tfr-flir_tfc-homologs.csv
```
You need to edit the `loops` and `plot` sections of this file before using the next script.

## Optimizing the rigid transformations between cameras and leddars

The final step is to use the [extrinsic_generic_loops.py](./extrinsic_generic_loops.py) to compute the transformations between the different sensors. For each pariwise transformation listed in the YAML file, the script will compute a first estimate of the transformation using a RANSAC with an SVD decomposition. Most ouliers will be removed by the RANSAC algorithm and only the inliers will be kept for the remainder of the optimization process. Once we have a first estimate of each transformation we optimize transformation loops as defined in the `loops` section. The loops are defined as a series of transformations that loop back to the initial reference frame. For example:
```YAML
'flir_tfl-flir_tfr->flir_tfr-flir_tfc->-flir_tfl-flir_tfc'
```
this loop starts in the left camera, transforms the points to the right camera reference frame, then in the center camera and back to the left camera. The sequence is indicated with arrows `->` and inverse transforms are specified with a `-` prefix. All transforms listed in the loop must be in the `transformations` section.

The extrinsic calibration script is run like this:
```bash
$ python3 extrinsic_generic_loops.py /nas/path/to/loop_config.yml
```
By default it will create a pickle file for each pair of sensors.

If the `plot` section contains a `reference` element the script will also plot the reference frames of the different sensors with respect to this reference sensor. Also, it will plot the homolog points transformed back to this reference frame. For example
```yaml
plot:
  reference: flir_tfl
```
will draw everything with respect to the left camera reference frame.

## Using the extrinsic calibration in your platform.yml

All sensors listed in a `platform.yml` file have an `extrinsics` section. This section should map to the folder containing the pickle files created by the `extrinsic_generic_loops.py` script. Facilities are provided by the API to transform point clouds from one sensor to the other and to back project the points in a camera image, see `das.api.samples.Echo.point_cloud` and `das.api.samples.Image.project_pts`.





