import numpy as np


def data_files(directory):
    print(f'+ {directory}')
    paths = []
    for path in sorted(directory.rglob('*.data')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')
        paths.append(path)
    return paths


def list_folder_names(path):
    return [e.name for e in path.iterdir() if e.is_dir()]


def list_file_names(path, ext=None):
    if ext:
        return [
            e.name for e in path.iterdir() if e.is_file() and e.suffix == ext
        ]
    else:
        return [e.name for e in path.iterdir() if e.is_file()]


def list_folders(path):
    return [e for e in path.iterdir() if e.is_dir()]


def list_files(path, ext=None):
    if ext:
        return [e for e in path.iterdir() if e.is_file() and e.suffix == ext]
    else:
        return [e for e in path.iterdir() if e.is_file()]


def load_from_bin_file(path):
    if path.suffix != ".bin":
        raise AssertionError("Files must be in binary format.")
    data = np.fromfile(path, dtype=np.int32)
    num_channels = 32
    return np.array(data.reshape(-1, num_channels + 4).T, dtype=np.float)


def load_from_file(filepath, nch, dtype, order="C"):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape(-1, nch).T


def tree(directory, ignore=[]):
    ignore_files = [".DS_Store"] + ignore
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):

        if path not in ignore_files:
            depth = len(path.relative_to(directory).parts)
            spacer = '    ' * depth
            print(f'{spacer}+ {path.name}')