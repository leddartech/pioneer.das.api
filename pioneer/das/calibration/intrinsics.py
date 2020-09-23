#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from pioneer.das.api.sources import filesource

from functools import partial
from io import BytesIO
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue
from multiprocessing.queues import Empty
from tqdm import tqdm

import cv2
import math
import multiprocessing
import numpy
import os
import pickle
import time

def find_chessboard(gray, pattern_size, checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE):
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags = checkerboard_flags)

    if found:
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    return found, corners

def find_chessboard_proc(input_queue, output_queue, pattern_size):
    while True:
        try:
            data = input_queue.get(block=False, timeout=0.1)
            if data == 'QUIT':
                break
            else:
                i, image = data
                out = find_chessboard(image, pattern_size)
                output = [data, out]
                output_queue.put(output)
        except Empty:
            pass

class ChessboardFinderProc(object):

    """Finds a chessboard inside a list of image files. Optionally uses a
    multiprocessing pool of workers.
    """

    def __init__(self, workers=0):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = max(1, workers)
        self.procs = []

    def __call__(self, images, pattern_size):

        if self.workers == 1:
            corners_list = []
            for image in images:
                found, corners = find_chessboard(image, pattern_size)
                corners_list.append(corners if found else None)
            return corners_list

        for _ in range(self.workers):
            proc = Process(target=find_chessboard_proc,
                args=(self.input_queue, self.output_queue, pattern_size))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

        # slice the file list in into list or `workers` elements
        n = len(images)
        for i, image in enumerate(images):
            self.input_queue.put((i, image))

        count = 0
        with tqdm(total=n) as pbar:
            while count < n:
                c = self.output_queue.qsize()
                dt = c - count
                pbar.update(dt)
                count = c
                time.sleep(0.2)

        for _ in range(self.workers):
            self.input_queue.put('QUIT')

        corners_list = [None] * n
        # IMPORTANT empty the output queue before joining
        for _ in range(n):
            (i, image), (found, corners) = self.output_queue.get()
            if found:
                corners_list[i] = corners
            else:
                corners_list[i] = None

        assert self.output_queue.qsize() == 0

        for p in self.procs:
            p.join(timeout=0.2)
            p.terminate()

        self.procs = []

        return corners_list


def make_object_points(pattern_size, dx, dy = None):

    n_cols, n_rows = pattern_size
    if dy is None:
        dy = dx

    points = numpy.zeros((n_cols * n_rows, 3), dtype=numpy.float32)
    points[:, :2] = numpy.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    points[:, 0] *= dy
    points[:, 1] *= dx

    return points

def calibrate_camera(imgpoints, object_points_3d, image_size,  cameraMatrixGuess = None, flags = 0):
    objpoints = [] # 3d point in real world space

    filtered = []
    for im in imgpoints:
        if im is not None:
            filtered.append(im)
            objpoints.append(object_points_3d)

    if cameraMatrixGuess is not None:
        cameraMatrixGuess = numpy.copy(cameraMatrixGuess)

    reprojectionError\
    , cameraMatrix\
    , distCoeffs\
    , rvecs, tvecs\
    , stdDeviationsIntrinsics, stdDeviationsExtrinsics\
    , perViewErrors = cv2.calibrateCameraExtended(objpoints, filtered, image_size, cameraMatrixGuess, None, flags = flags)

    return {  'matrix' : cameraMatrix, 'distortion' : distCoeffs
            , 'rvecs' : rvecs, 'tvecs' : tvecs  
            , 'reprojectionError': reprojectionError
            , 'stdDeviationsIntrinsics' : stdDeviationsIntrinsics
            , 'stdDeviationsExtrinsics' : stdDeviationsExtrinsics
            , 'perViewErrors' : perViewErrors}

def calibrate_camera_stereo(imgpoints_left, imgpoints_right
, object_points_3d, image_size
, matrix_left, distortion_left
, matrix_right, distortion_right
, criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)
, flags = cv2.CALIB_FIX_INTRINSIC):
    objpoints = [] # 3d point in real world space

    filtered_left = []
    filtered_right = []
    for i, im in enumerate(imgpoints_left):
        if im is not None and imgpoints_right[i] is not None:
            filtered_left.append(im)
            filtered_right.append(imgpoints_right[i])
            objpoints.append(object_points_3d)

    T = numpy.zeros((3, 1), dtype=numpy.float64)
    R = numpy.eye(3, dtype=numpy.float64)
    reprojectionError, matrix_left, distortion_left, matrix_right, distortion_right, R, T, E, F, perViewErrors = \
    cv2.stereoCalibrateExtended(objpoints, filtered_left, filtered_right,
                                matrix_left, distortion_left,
                                matrix_right, distortion_right,
                                image_size,                        
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5),
                                flags = flags)

    return {  'R' : R, 'T' : T, 'E' : E, 'F' : F
            , 'reprojectionError': reprojectionError
            , 'perViewErrors' : perViewErrors
            , 'matrix_left' : matrix_left, 'distortion_left' : distortion_left
            , 'matrix_right': matrix_right, 'distortion_right' : distortion_right}

def compute_progress(all_params, param_names = ['x', 'y', 'size', 'skew'], param_ranges = [0.7, 0.7, 0.4, 0.5]):

    # Find range of checkerboard poses covered by samples in database
    min_params = all_params[0]
    max_params = all_params[0]
    for params in all_params[1:]:
        min_params = _lmin(min_params, params)
        max_params = _lmax(max_params, params)
    # Don't reward small size or skew
    min_params = [min_params[0], min_params[1], 0., 0.]

    # For each parameter, judge how much progress has been made toward adequate variation
    progress = [min((hi - lo) / r, 1.0) for (lo, hi, r) in zip(min_params, max_params, param_ranges)]

    return list(zip(param_names, min_params, max_params, progress))

def is_good_sample(params, params_list):
    """
    Returns true if the checkerboard detection described by params should be added to the database.
    """
    if not params_list:
        return True

    def param_distance(p1, p2):
        return sum([abs(a-b) for (a,b) in zip(p1, p2)])

    d = min([param_distance(params, p) for p in params_list])

    return d > 0.2

def get_parameters(corners, pattern_size, image_size):
    """
    Return list of parameters [X, Y, size, skew] describing the checkerboard view.
    """
    (width, height) = image_size
    Xs = corners[:,:,0]
    Ys = corners[:,:,1]
    area = _get_area(corners, pattern_size)
    border = math.sqrt(area)
    # For X and Y, we "shrink" the image all around by approx. half the board size.
    # Otherwise large boards are penalized because you can't get much X/Y variation.
    p_x = min(1.0, max(0.0, (numpy.mean(Xs) - border / 2) / (width  - border)))
    p_y = min(1.0, max(0.0, (numpy.mean(Ys) - border / 2) / (height - border)))
    p_size = math.sqrt(area / (width * height))
    skew = _get_skew(corners, pattern_size)
    params = [p_x, p_y, p_size, skew]
    return params

def downsample_and_detect(img, pattern_size):
    """
    Downsample the input image to approximately VGA resolution and detect the
    calibration target corners in the full-size image.

    Combines these apparently orthogonal duties as an optimization. Checkerboard
    detection is too expensive on large images, so it's better to do detection on
    the smaller display image and scale the corners back up to the correct size.

    Returns (scrib, corners, downsampled_corners, board, (x_scale, y_scale)).
    """
    # Scale the input image down to ~VGA size
    height = img.shape[0]
    width = img.shape[1]
    scale = math.sqrt( (width*height) / (640.*480.) )
    if scale > 1.0:
        scrib = cv2.resize(img, (int(width / scale), int(height / scale)))
    else:
        scale = 1.0
        scrib = img
    # Due to rounding, actual horizontal/vertical scaling may differ slightly
    x_scale = float(width) / scrib.shape[1]
    y_scale = float(height) / scrib.shape[0]

    # Detect checkerboard
    (ok, downsampled_corners) = _get_corners(scrib, pattern_size, refine = True)

    # Scale corners back to full size image
    corners = None
    if ok:
        if scale >= 1.0:
            # Refine up-scaled corners in the original full-res image
            # TODO Does this really make a difference in practice?
            corners_unrefined = downsampled_corners.copy()
            corners_unrefined[:, :, 0] *= x_scale
            corners_unrefined[:, :, 1] *= y_scale
            radius = max(7, int(math.ceil(scale)))
            if len(img.shape) == 3 and img.shape[2] == 3:
                mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                mono = img
            cv2.cornerSubPix(mono, corners_unrefined, (radius,radius), (-1,-1),
                                          ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ))
            corners = corners_unrefined
        else:
            corners = downsampled_corners

    return (scrib, corners, downsampled_corners, (x_scale, y_scale))

def save_as_datasource(images, path, sensor = 'flir', pos = 'tfl', calib = None, cam_specs = None, pattern_spces = None):

    output_datasouce_name = '{}_{}_intr-calib-img.zip'.format(sensor, pos)

    if not os.path.exists(path):
        os.makedirs(path)

    with filesource.ZipFileWriter(os.path.join(path, output_datasouce_name)) as f:
        for i, image in enumerate(images):
            f.write_array_as_numbered_png(i, image)

        f.write_array_as_txt('timestamps.csv', numpy.arange(len(images), dtype='u8'), fmt='%d')

    output_file = os.path.join(path, '{}_{}_calib-results.pkl'.format(sensor, pos))
    with open(output_file, 'wb') as f:
        pickle.dump(calib, f)

    output_file = os.path.join(path, '{}_{}_calib-specs.pkl'.format(sensor, pos))
    with open(output_file, 'wb') as f:
        pickle.dump({'cam-specs': cam_specs, 'pattern-specs' : pattern_spces}, f)


###################
# Helper functions:

def _lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]

def _lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def _get_outside_corners(corners, pattern_size):
    """
    Return the four corners of the board as a whole, as (up_left, up_right, down_right, down_left).
    """
    xdim, ydim = pattern_size

    if corners.shape[1] * corners.shape[0] != xdim * ydim:
        raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" % (corners.shape[1] * corners.shape[0],
                                                                                 xdim, ydim))

    up_left    = corners[0,0]
    up_right   = corners[xdim - 1,0]
    down_right = corners[-1,0]
    down_left  = corners[-xdim,0]

    return (up_left, up_right, down_right, down_left)

def _get_skew(corners, pattern_size):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust, outside corners occasionally
    # get mis-detected
    up_left, up_right, down_right, _ = _get_outside_corners(corners, pattern_size)

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return math.acos(numpy.dot(ab,cb) / (numpy.linalg.norm(ab) * numpy.linalg.norm(cb)))

    skew = min(1.0, 2. * abs((math.pi / 2.) - angle(up_left, up_right, down_right)))
    return skew

def _get_area(corners, pattern_size):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right, down_left) = _get_outside_corners(corners, pattern_size)
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.

def _get_corners(img, pattern_size, refine=True, checkerboard_flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE, reorder=False):
    """
    Get corners for a particular chessboard for an image
    """
    n_cols, n_rows = pattern_size
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img
    (ok, corners) = cv2.findChessboardCorners(mono, pattern_size, flags = checkerboard_flags)
    if not ok:
        return (ok, corners)

    # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
    # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
    # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras
    BORDER = 8
    if not all([(BORDER < corners[i, 0, 0] < (w - BORDER)) and (BORDER < corners[i, 0, 1] < (h - BORDER)) for i in range(corners.shape[0])]):
        ok = False

    if reorder:
        # Ensure that all corner-arrays are going from top to bottom.
        if n_rows!=n_cols:
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = numpy.copy(numpy.flipud(corners))
        else:
            direction_corners=(corners[-1]-corners[0])>=numpy.array([[0.0,0.0]])

            if not numpy.all(direction_corners):
                if not numpy.any(direction_corners):
                    corners = numpy.copy(numpy.flipud(corners))
                elif direction_corners[0][0]:
                    corners=numpy.rot90(corners.reshape(n_rows,n_cols,2)).reshape(n_cols*n_rows,1,2)
                else:
                    corners=numpy.rot90(corners.reshape(n_rows,n_cols,2),3).reshape(n_cols*n_rows,1,2)

    if refine and ok:
        # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
        # correct corner, but not so large as to include a wrong corner in the search window.
        min_distance = float("inf")
        for row in range(n_rows):
            for col in range(n_cols - 1):
                index = row*n_rows + col
                min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
        for row in range(n_rows - 1):
            for col in range(n_cols):
                index = row*n_rows + col
                min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + n_cols, 0]))
        radius = int(math.ceil(min_distance * 0.5))
        cv2.cornerSubPix(mono, corners, (radius,radius), (-1,-1),
                                      ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ))

    return (ok, corners)



