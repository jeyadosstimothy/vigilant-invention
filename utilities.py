import os
from distutils.dir_util import copy_tree


CHECKPOINT_DIRECTORY = 'checkpoints'


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_checkpoint_file(dataset_name, network_name, is_dir=False):
    dataset_directory = os.path.join(CHECKPOINT_DIRECTORY, dataset_name)
    directory = os.path.join(dataset_directory, network_name)
    create_directory(directory)
    if is_dir:
        return directory
    return os.path.join(directory, 'ep{epoch:03d}-{val_acc:.3f}.hdf5')


def copy_and_overwrite(from_path, to_path):
    copy_tree(from_path, to_path)
