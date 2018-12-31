from abc import ABC, abstractmethod
from keras import datasets, utils
import numpy as np
import cv2


def resize_images(dataset, dim=(32, 32)):
    return np.array([cv2.resize(img, dim) for img in dataset])

class Dataset(ABC):
    train_x = None
    train_y = None
    test_x = None
    test_y = None

    @abstractmethod
    def __init__(self):
        pass

    @property
    def name(self):
        return type(self).__name__

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def instance_shape(self):
        pass

# Keras datasets can be found here: /home/$USER/.keras/datasets
class Mnist(Dataset):
    def __init__(self):
        (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        train_x = train_x.reshape(*train_x.shape, 1)
        test_x = test_x.reshape(*test_x.shape, 1)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10

    @property
    def instance_shape(self):
        return self.train_x.shape[1:]


class FashionMnist(Dataset):
    def __init__(self):
        (train_x, train_y), (test_x, test_y) = datasets.fashion_mnist.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        train_x = train_x.reshape(*train_x.shape, 1)
        test_x = test_x.reshape(*test_x.shape, 1)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10

    @property
    def instance_shape(self):
        return self.train_x.shape[1:]


class Cifar10(Dataset):
    def __init__(self):
        (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10

    @property
    def instance_shape(self):
        return self.train_x.shape[1:]


class Cifar100(Dataset):
    def __init__(self):
        (train_x, train_y), (test_x, test_y) = datasets.cifar100.load_data(label_mode='fine')
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 100

    @property
    def instance_shape(self):
        return self.train_x.shape[1:]

