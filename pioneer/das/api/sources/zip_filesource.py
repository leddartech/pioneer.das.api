import pioneer.common.constants as Constants
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.sources.filesource import FileSource, try_all_patterns
from pioneer.das.api.loaders import pickle_loader

from ruamel.std import zipfile

import multiprocessing
import numpy as np
import pandas as pd
import os
import re
import sys
import threading
import time
import traceback
import yaml

class ZipFileSource(FileSource):
    """Loads a list of files from a zip archive."""

    def __init__(self, path, pattern=None, sort=True, loader=None,
                 check_timestamps=True, mode="lock"):
        super(ZipFileSource, self).__init__(path, pattern, sort, loader)
        self.path = path
        self.member_cached = None
        
        self.lock = multiprocessing.RLock()
        self.tlock = threading.RLock()
        self.archive = None
        self.set_mode(mode)

        self.nb_data_per_pkl_file = 1

        self.files, self.time_of_issues, self.timestamps, self.nb_data_per_pkl_file = self.get_files(self.access)

        if check_timestamps:
            self._check_timestamps_consistency()

    def get_files(self, access):

        try:
            members = access( lambda archive: archive.infolist())
        except Exception as e:
            traceback.print_exc()
            raise e

        files = []

        timestamps = None
        time_of_issues = None
        nb_data_per_pkl_file = 1

        for m in members:
            name = m.filename
            if self.pattern is None:
                match, self.pattern = try_all_patterns(name)
            else:
                match = re.match(self.pattern, name)
            if match:
                groups = match.groups()
                if self.sort and not groups:
                    raise ValueError('no groups')
                if self.sort and groups:
                    sample = (int(groups[0]), m)
                else:
                    sample = m
                files.append(sample)

            # read config for multiple rows per .pkl file (high fps sensors)
            elif(name == Constants.CONFIG_YML_PATTERN):
                def f_CONFIG_YML_PATTERN(archive):
                    with archive.open(name) as stream:
                        data_yaml = yaml.safe_load(stream)
                        return data_yaml['nb_vectors_in_pkl']
                
                nb_data_per_pkl_file = access(f_CONFIG_YML_PATTERN)

            # read timestamps
            elif(name == Constants.TIMESTAMPS_CSV_PATTERN):
                def f_TIMESTAMPS_CSV_PATTERN(archive):
                    with archive.open(name) as stream:
                        # sensor_ts = np.loadtxt(stream, dtype='u8', delimiter=' ', ndmin=2)
                        sensor_ts = pd.read_csv(stream, delimiter=" ", dtype='u8', header=None).values

                        timestamps = sensor_ts[:,0]
                        # check if ts is always go up, for imu data (more than 1 data per pkl file) the test is not complet -> to improve
                        if len(timestamps)>2 and (np.min(np.diff(timestamps.astype(np.int64)))<0):
                            LoggingManager.instance().warning('Timestamps are not strictly increasing for datasource file {}'.format(self.path))

                        if sensor_ts.shape[1] > 1:
                            time_of_issues = sensor_ts[:,1]
                        else:
                            time_of_issues = timestamps
                        return time_of_issues, timestamps

                time_of_issues, timestamps = access(f_TIMESTAMPS_CSV_PATTERN)
         
        if self.sort:
            files.sort()
            files = [s[1] for s in files]
            
        return files, time_of_issues, timestamps, nb_data_per_pkl_file


    def set_mode(self, mode):

        def access_file(f):
            with zipfile.ZipFile(self.path, 'r') as archive:
                return f(archive)

        def access_raw(f):
            return f(self.archive)

        def access_lock(f):
            exception = None
            while True:
                try: # workaround for BadZipFile in multiprocessing mode
                    with self.lock:
                        with self.tlock:
                            return f(self.archive)
                except Exception as e:
                    if False:
                        sys.stdout.write(f"\rZipFileSource {os.path.basename(self.path)} read error, falling back to 'file' mode{' ' * 10}")
                        sys.stdout.flush()
                    try:
                        if self.archive is not None:
                            self.archive.close()
                            self.archive = None
                        self.files, self.time_of_issues, self.timestamps, self.nb_data_per_pkl_file = self.get_files(access_file)
                        rv =  access_file(f)
                        self.archive = zipfile.ZipFile(self.path, 'r')
                        return rv

                    except Exception as access_file_e:
                        self.archive = zipfile.ZipFile(self.path, 'r')
                        print(f'\rZipFileSource fallback failed: {access_file_e}, retrying infinitely')
                        time.sleep(0.1)

        if mode == "lock":
            self.archive = zipfile.ZipFile(self.path, 'r')
            self.access = access_lock
        elif mode == "file":
            self.archive = None
            self.access = access_file
        elif mode == "none":
            self.archive = zipfile.ZipFile(self.path, 'r')
            self.access = access_raw

    def __del__(self):
        if self.archive is not None:
            self.archive.close()

    def __len__(self):
        return self.timestamps.shape[0]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f'For datasource {os.path.basename(self.path)} index {index} >= len {len(self)}')
        if self.nb_data_per_pkl_file == 1:
            member = self.files[index]
            data_bytes = self.access( lambda archive: archive.read(member))
            return self.loader(data_bytes)
        # pkl containing nb_data_per_pkl_file data rows
        else:
            if(index<0):
                index = len(self) + index
                if(index < 0):
                    raise IndexError('index {} < 0'.format(index))
            member = self.files[index//self.nb_data_per_pkl_file]
            # no reload
            if self.member_cached is None or self.member_cached!=member:
                data_bytes = self.access( lambda archive: archive.read(member))
                self.data_cached = self.loader(data_bytes)
                self.member_cached=member

            return self.data_cached[index%self.nb_data_per_pkl_file] # compatible with array and struct array

    def _get_nb_timestamps_and_files_or_rows(self):
        # nb timestamps
        nts = len(self)
        # nb files, or rows
        if(self.nb_data_per_pkl_file == 1):
            nfiles = len(self.files)
        else:
            # if empty no data
            if len(self.files) == 0:
                nfiles = 0
            # else, (N-1) x nb_data_per_pkl_file + nb rows in last file
            else:
                data_bytes = self.access(lambda archive: archive.read(self.files[-1]))
                nfiles = (len(self.files)-1)*self.nb_data_per_pkl_file + self.loader(data_bytes).shape[0]
        return nts, nfiles


    def _check_timestamps_consistency(self):
        nts, nfiles = self._get_nb_timestamps_and_files_or_rows()

        if nfiles != nts:
            n = min(nts, nfiles)
            LoggingManager.instance().warning('The number of timestamps and data files are '
                            'different for sensor %s (nfiles: %d != nts: %d). '
                            'Keeping the %d first timestamps and files'
                            %(self.path, nfiles, nts, n))
            self.timestamps = self.timestamps[:n]

            if(self.nb_data_per_pkl_file == 1):
                self.files = self.files[:n]
            else:
                if n%self.nb_data_per_pkl_file == 0:
                    self.files = self.files[:int(n/self.nb_data_per_pkl_file)]
                else:
                    # on va conserver un fichier à la fin qui ne sera pas utilisé en entier
                    self.files = self.files[:n//self.nb_data_per_pkl_file+1]

            nts, nfiles = self._get_nb_timestamps_and_files_or_rows()
            assert nfiles == nts

    def get(self, name, loader=None):
        if loader is None:
            loader = pickle_loader
        try:
            data_bytes = self.access(lambda archive: archive.read(archive.getinfo(name)))
            data = loader(data_bytes)
        except KeyError:
            data = None
        return data