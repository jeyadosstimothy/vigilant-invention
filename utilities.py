import os


CHECKPOINT_DIRECTORY = 'checkpoints'


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_checkpoint_file(dataset_name, network_name):
    dataset_directory = os.path.join(CHECKPOINT_DIRECTORY, dataset_name)
    directory = os.path.join(dataset_directory, network_name)
    create_directory(directory)
    return os.path.join(directory, 'ep{epoch:03d}-{val_acc:.3f}.hdf5')
