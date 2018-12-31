import os


CHECKPOINT_DIRECTORY = 'checkpoints'


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_checkpoint_file(dataset_name):
    dataset_directory = os.path.join(CHECKPOINT_DIRECTORY, dataset_name)
    create_directory(dataset_directory)
    return os.path.join(dataset_directory, 'epoch{epoch:03d}-{val_acc:.3f}.hdf5')
