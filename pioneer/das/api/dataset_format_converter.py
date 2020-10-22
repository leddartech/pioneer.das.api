import glob
import multiprocessing
import os
import shutil
import tqdm
import zipfile



def __extract_or_copy_file(args):
    filepath, output = args
    try:
        if filepath.endswith('.zip'):
            filename = filepath.split('/')[-1].split('.')[0]
            archive = zipfile.ZipFile(filepath, 'r')
            archive.extractall(f'{output}/{filename}/.')
        elif os.path.isdir(filepath):
            name = filepath.split('/')[-1]
            shutil.copytree(filepath, f'{output}/{name}')
        else:
            name = filepath.split('/')[-1]
            shutil.copyfile(filepath, f'{output}/{name}')
    except:
        print(f'Failed to extract or copy: {filepath}')

def extract_zipfiles(dataset:str, output:str, files:list='all'):
    """Make a copy of a dataset with unzipped files
    
        Args:
            dataset (str): Path to the original dataset
            output (str): Path to the copy to be created. Must be non-existant for safety reasons.
            files (list): list of zipfile names to include. Default is 'all'.
    """

    if not os.path.exists(output):
        os.mkdir(output)
    else:
        print(f"Cannot extract to {output}, because it is an already existing directory.")
        return

    all_files = glob.glob(f'{dataset}/*')
    
    if isinstance(files, list):
        all_files = [f for f in all_files if f.split('/')[-1] in files]

    all_files = [[f, output] for f in all_files]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(__extract_or_copy_file, all_files), total = len(all_files)):
        pass