# Pierre Merriaux, Guillaume Dumont
# Novembre 2018
# permet de detecter les mires dans dans une liste d'image
# puis de faire la calibation intrinsec de la camera

from __future__ import division, print_function

from pioneer.das.api import chessboard, platform
from pioneer.das.api.sources import filesource
import argparse
import copy
import glob
import multiprocessing
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Process, Queue
from multiprocessing.queues import Empty

import cv2
import numpy as np
from tqdm import tqdm




def find_chessboard(img, pattern_size, slow=False):
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if slow:
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
    else:
        flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if found:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        # cv2.drawChessboardCorners(img, pattern_size, corners, found)
        # if draw:
        #     output_filename = os.path.join(output_folder,
        #         os.path.basename(image_file) + '.jpg')
        #     cv2.imwrite(output_filename, img)

    return found, corners, gray.shape[::-1]

def find_chessboard_proc(dataset_folder, search_string, input_queue, output_queue, pattern_size, slow=False):
    plat = platform.Platform(dataset_folder)
    source = plat[search_string]
    while True:
        try:
            index = input_queue.get(block=False, timeout=0.1)
            if index == 'QUIT':
                break
            else:
                data = source[index]
                image = data.raw
                out = find_chessboard(image, pattern_size, slow)
                output = [index, out]
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

    def __call__(self, plat, search_string, pattern_size,
                 slow=False):
        output_dict = dict()

        dataset_folder = plat.dataset
        files = plat[search_string]
        n = len(files)
        for index in range(n):
            self.input_queue.put(index)

        for _ in range(self.workers):
            proc = Process(target=find_chessboard_proc,
                args=(dataset_folder, search_string, self.input_queue, self.output_queue, pattern_size, slow))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

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

        # IMPORTANT empty the output queue before joining
        corners_dict = {}
        for _ in range(n):
            index, (found, corners, shape) = self.output_queue.get()
            output_dict.setdefault('shape', shape)
            corners_dict[index] = dict(found=found, corners=corners)

        output_dict['corners'] = corners_dict

        assert self.output_queue.qsize() == 0

        for p in self.procs:
            p.join(timeout=0.2)
            p.terminate()

        self.procs = []

        return output_dict

class Intrinsics(object):
    def __init__(self, datasetFolder, workers=0):
        self.datasetFolder = datasetFolder
        self.plat = platform.Platform(datasetFolder)
        self.chessboard_specs = chessboard.load_chessboard_specifications(self.plat)
        self.nx = self.chessboard_specs['nx']
        self.ny = self.chessboard_specs['ny']
        self.chessboard_points = self.chessboard_specs['points']
        self.finder = ChessboardFinderProc(workers)

    def processChessboardSearch(self, searchString, slow):
        print('Processing chessboards with {} worker processes'.format(self.finder.workers))

        if not os.path.exists(self.datasetFolder):
            raise FileNotFoundError(self.datasetFolder)

        if not os.path.exists(os.path.join(self.datasetFolder, 'platform.yml')):
            raise FileNotFoundError(os.path.join(self.datasetFolder, 'platform.yml'))

        dic = self.finder(self.plat, searchString,
            (self.nx, self.ny), slow)
        dic['chessboard'] = self.chessboard_specs

        sensor, pos, _ = platform.parse_datasource_name(searchString)

        output_datasouce_name = '{}_{}_chessboards.zip'.format(sensor, pos)
        with filesource.ZipFileWriter(os.path.join(self.datasetFolder, output_datasouce_name)) as f:
            f.write_numbered_pickle(0, dic)
            f.write_array_as_txt('timestamps.csv', np.array([0], dtype='u8'), fmt='%d')

    def computeIntrinsicFiltered(self, chessboard_zip_name):
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        plat = self.plat
        sensor, pos, _ = platform.parse_datasource_name(chessboard_zip_name)
        file_list = plat[chessboard_zip_name]
        dic = file_list[0].raw

        chessboard_specs = dic['chessboard']
        corners_dict = dic['corners']
        shape = dic['shape']

        for key, value in corners_dict.items():
            try:
                corners = value['corners']
                found = value['found']
                if not found:
                    continue
            except:
                continue

            imgpoints.append(corners)
            objpoints.append(chessboard_specs['points'])

        imgpoints = np.stack(imgpoints)
        objpoints = np.stack(objpoints)

        # à remettre correctement une fois les pickles bien enregistré
        _, matrix, distorsion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        #ret, matrix, distorsion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, dic['shape'], None, None)

        output_folder = os.path.join(self.datasetFolder, 'intrinsic_calibration')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file = os.path.join(output_folder,
                                   '{}_{}.pkl'.format(sensor, pos))
        with open(output_file, 'wb') as f:
            pickle.dump({'matrix':matrix,
                         'distortion': distorsion,
                         'camera_tf': [rvecs, tvecs]}, f)


    def filterChessboard(self, searchString, threshold):
        print(searchString, threshold)
        sensor, pos, _ = platform.parse_datasource_name(searchString)
        plat = self.plat
        dic = plat[searchString][0].raw
        newDic = copy.deepcopy(dic)
        newDic['corners'] = {}

        corners_dict = dic['corners']
        for key, items in corners_dict.items():
            found = items['found']
            if not found:
                continue

            corners = items['corners']

            keep = True
            for itemsNew in newDic['corners'].values():
                cornersNew = itemsNew['corners']
                #print(keys,keysNew)
                r = np.squeeze(cornersNew) - np.squeeze(corners)
                #print("np.sqrt(r*r)", r, np.sqrt(np.sum(r*r)))
                if np.sqrt(np.sum(r*r))<threshold:
                    keep=False
                    break
            if (keep):
                newDic['corners'][key] = dict(found=True, corners=corners)

        fname = os.path.join(self.datasetFolder,
            '{}_{}_chessboards-filtered.zip'.format(sensor, pos))

        with filesource.ZipFileWriter(fname) as f:
            f.write_numbered_pickle(0, newDic)
            f.write_array_as_txt('timestamps.csv', np.array([0], dtype='u8'), fmt='%d')

        print("keep {}/{} chessboards".format(len(newDic['corners']),len(dic['corners'])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process the flir camera intrinsic calibration, from exported rtmaps dataset.')

    parser.add_argument('datasetFolder', default=None)
    parser.add_argument('zipfileName', default=None, type=str, help='Zip file name')
    parser.add_argument('-t', default=100, type=float, help='Min pixel distance threshold to keep the chessboard')
    parser.add_argument('--filter', action='store_true', help='Filter the chessboards')
    parser.add_argument('--chessboard', action='store_true', help='compute the chessboards')
    parser.add_argument('--slow', action='store_true', default=False,
        help='Use default findChessboard options '
             '(makes the chessboard detection slower but more accurate)')
    parser.add_argument('--intrinsic', action='store_true', help='compute intrinsic matrix of filtered chessboard')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-2),
        help='Number of worker processes to use to detect the chessboard')
    # parser.add_argument('--nx', default=3, type=int, help='x chessboard intersection number')
    # parser.add_argument('--ny', default=4, type=int, help='y chessboard intersection number')
    # parser.add_argument('--dx', default=0.145, type=float, help='chessboard case x size')
    # parser.add_argument('--dy', default=0.145, type=float, help='chessboard case y size')
    args = parser.parse_args()

    zipfile, _ = os.path.splitext(os.path.split(args.zipfileName)[1])

    if args.chessboard:
        intr = Intrinsics(args.datasetFolder,
            workers=args.workers)
        intr.processChessboardSearch(zipfile, args.slow)

    if args.filter:
        intr = Intrinsics(args.datasetFolder)
        intr.filterChessboard(zipfile, args.t)

    if args.intrinsic:
        intr = Intrinsics(args.datasetFolder)
        intr.computeIntrinsicFiltered(zipfile)
