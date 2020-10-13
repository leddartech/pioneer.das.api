import pioneer.common.constants as Constants
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.sources.filesource import FileSource, try_all_patterns
from pioneer.das.api.loaders import pickle_loader

import numpy as np
import pandas as pd
import os
import re
import yaml
import glob


class DirSource(FileSource):
    """Loads a list of files from a directory."""

    def __init__(self, path, pattern=None, sort=True, loader=None, check_timestamps=True):
        super(DirSource, self).__init__(path, pattern, sort, loader)
        self.path = path
        self.nb_data_per_pkl_file = 1
        self.member_cached = None

        self.files, self.time_of_issues, self.timestamps, self.nb_data_per_pkl_file = self.get_files()

        if check_timestamps:
            self._check_timestamps_consistency()

    def get_files(self):

        files = []

        timestamps = None
        time_of_issues = None
        nb_data_per_pkl_file = 1

        for fullpath in glob.glob(f'{self.path}/*'):
            name = fullpath.split('/')[-1]
            if self.pattern is None:
                match, self.pattern = try_all_patterns(name)
            else:
                match = re.match(self.pattern, name)
            if match:
                groups = match.groups()
                if self.sort and not groups:
                    raise ValueError('no groups')
                if self.sort and groups:
                    sample = (int(groups[0]), fullpath)
                else:
                    sample = fullpath
                files.append(sample)

            # read config for multiple rows per .pkl file (high fps sensors)
            elif(name == Constants.CONFIG_YML_PATTERN):
                with open(fullpath) as f:
                    data_yaml = yaml.safe_load(f)
                    nb_data_per_pkl_file = data_yaml['nb_vectors_in_pkl']
                
            # read timestamps
            elif(name == Constants.TIMESTAMPS_CSV_PATTERN):
                with open(fullpath) as f:
                    sensor_ts = pd.read_csv(f, delimiter=" ", dtype='u8', header=None).values

                    timestamps = sensor_ts[:,0]
                    # check if ts is always go up, for imu data (more than 1 data per pkl file) the test is not complet -> to improve
                    if len(timestamps)>2 and (np.min(np.diff(timestamps.astype(np.int64)))<0):
                        LoggingManager.instance().warning('Timestamps are not strictly increasing for datasource file {}'.format(self.path))

                    if sensor_ts.shape[1] > 1:
                        time_of_issues = sensor_ts[:,1]
                    else:
                        time_of_issues = timestamps
        
        if self.sort:
            files.sort()
            files = [s[1] for s in files]
            
        return files, time_of_issues, timestamps, nb_data_per_pkl_file

    def __del__(self):
        pass

    def __len__(self):
        return self.timestamps.shape[0]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f'For datasource {os.path.basename(self.path)} index {index} >= len {len(self)}')
        if self.nb_data_per_pkl_file == 1:
            return self.loader(self.files[index])
        else:
            if(index<0):
                index = len(self) + index
                if(index < 0):
                    raise IndexError('index {} < 0'.format(index))
            member = self.files[index//self.nb_data_per_pkl_file]
            # no reload
            if self.member_cached is None or self.member_cached!=member:
                self.data_cached = self.loader(member)
                self.member_cached=member

            return self.data_cached[index%self.nb_data_per_pkl_file] # compatible with array and struct array

    def _get_nb_timestamps_and_files_or_rows(self):
        if(self.nb_data_per_pkl_file == 1):
            nfiles = len(self.files)
        else:
            if len(self.files) == 0:
                nfiles = 0
            else:
                nfiles = (len(self.files)-1)*self.nb_data_per_pkl_file + self.loader(self.files[-1]).shape[0]
        return len(self), nfiles

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
            data = loader(name)
        except KeyError:
            data = None
        return data