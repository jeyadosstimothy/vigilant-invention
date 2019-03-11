from abc import ABC, abstractmethod
from tensorflow.keras import datasets, utils, backend as K
import numpy as np
import cv2, os, zipfile
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ENAS_Keras.src.utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def resize_image(img, dim=(32, 32)):
    return cv2.resize(img, dim)


def resize_images(dataset, dim=(32, 32)):
    return np.array([resize_image(img, dim) for img in dataset])


class Dataset(ABC):
    train_x = None
    train_y = None
    test_x = None
    test_y = None

    @abstractmethod
    def __init__(self, path):
        pass

    @property
    def name(self):
        return type(self).__name__

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    def instance_shape(self):
        return self.train_x.shape[1:]

    def get_data_gen(self):
        return ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))

    def get_data_flow_gen(self, batch_size):
        return MixupGenerator(
            self.train_x, self.train_y, batch_size=batch_size, alpha=0.2, datagen=self.get_data_gen())()


# TODO: subtract pixel mean to improve accuracy
# Keras datasets can be found here: /home/$USER/.keras/datasets
class Mnist(Dataset):
    def __init__(self, path='./'):
        (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        #train_x -= np.mean(train_x, axis=0, dtype=train_x.dtype)
        #test_x -= np.mean(test_x, axis=0, dtype=test_x.dtype)
        train_x = train_x.reshape(*train_x.shape, 1)
        test_x = test_x.reshape(*test_x.shape, 1)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        if K.image_data_format() == 'channels_first':
            train_x = train_x.reshape(train_x.shape[0], *reversed(train_x.shape[1:]))
            test_x = test_x.reshape(test_x.shape[0], *reversed(test_x.shape[1:]))
        else:
            train_x = train_x.reshape(*train_x.shape)
            test_x = test_x.reshape(*test_x.shape)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10


class FashionMnist(Dataset):
    def __init__(self, path='./'):
        (train_x, train_y), (test_x, test_y) = datasets.fashion_mnist.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        train_x = train_x.reshape(*train_x.shape, 1)
        test_x = test_x.reshape(*test_x.shape, 1)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        if K.image_data_format() == 'channels_first':
            train_x = train_x.reshape(train_x.shape[0], *reversed(train_x.shape[1:]))
            test_x = test_x.reshape(test_x.shape[0], *reversed(test_x.shape[1:]))
        else:
            train_x = train_x.reshape(*train_x.shape)
            test_x = test_x.reshape(*test_x.shape)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10


class Cifar10(Dataset):
    def __init__(self, path='./'):
        (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 10


class Cifar100(Dataset):
    def __init__(self, path='./'):
        (train_x, train_y), (test_x, test_y) = datasets.cifar100.load_data(label_mode='fine')
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 100


class SVHN(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path, 'SVHN')
        (train_x, train_y)  = self.load_data(os.path.join(os.path.join(self.path, 'train_32x32.mat')))
        (test_x, test_y)  = self.load_data(os.path.join(os.path.join(self.path, 'test_32x32.mat')))
        if K.image_data_format() == 'channels_first':
            train_x = train_x.reshape(train_x.shape[0], *reversed(train_x.shape[1:]))
            test_x = test_x.reshape(test_x.shape[0], *reversed(test_x.shape[1:]))
        else:
            train_x = train_x.reshape(*train_x.shape)
            test_x = test_x.reshape(*test_x.shape)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    def load_data(self, path):
        train_dict = sio.loadmat(self.path)
        X = np.asarray(train_dict['X'])
        X_t = []
        for i in range(X.shape[3]):
            X_t.append(X[:,:,:,i])
        X_t = np.asarray(X_t)

        Y_t = train_dict['y']
        for i in range(len(Y_t)):
            if Y_t[i]%10 == 0:
                Y_t[i] = 0
        return (X_t,Y_t)

    @property
    def num_classes(self):
        return 10


class GTSRB(Dataset):
    def __init__(self, path):
        self.path = os.path.join(path, 'GTSRB')
        (train_x, train_y), (test_x, test_y)  = self.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        if K.image_data_format() == 'channels_first':
            train_x = train_x.reshape(train_x.shape[0], *reversed(train_x.shape[1:]))
            test_x = test_x.reshape(test_x.shape[0], *reversed(test_x.shape[1:]))
        else:
            train_x = train_x.reshape(*train_x.shape)
            test_x = test_x.reshape(*test_x.shape)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    def load_data(self):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        train_x = [] # images
        train_y = [] # corresponding labels

        with zipfile.ZipFile(os.path.join(self.path, 'GTSRB_Final_Training_Images.zip')) as archive:
            prefix = 'GTSRB/Final_Training/Images/'
            for file in archive.namelist():
                if file.endswith('.ppm'):
                    c = int(file[len(prefix):len(prefix) + 5])
                    train_x.append(plt.imread(archive.open(file))) # the 1th column is the filename
                    train_y.append(c) # the 8th column is the label

        train_x, train_y = np.array(train_x), np.array(train_y)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)

        return (train_x, train_y), (test_x, test_y)

    @property
    def num_classes(self):
        return 43


class Flowers102(Dataset):
    def load_data(self):
        train_x = []
        train_y = sio.loadmat(os.path.join(self.path, 'imagelabels.mat'))['labels'][0] - 1

        image_dir = os.path.join(self.path, 'jpg')
        for file in os.listdir(image_dir):
            img = plt.imread(os.path.join(image_dir, file))
            train_x.append(resize_image(img))

        train_x = np.array(train_x)

        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)
        return (train_x, train_y), (test_x, test_y)

    def __init__(self, path):
        self.path = os.path.join(path, 'flowers-102')
        (train_x, train_y),(test_x, test_y)  = self.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 102


class Flowers(Dataset):
    def load_data(self):
        train_x = []
        train_y = []

        for file in os.listdir(self.path):
            classfolder = os.path.join(self.path, file)
            if os.path.isdir(classfolder):
                for image in os.listdir(classfolder):
                    img = plt.imread(os.path.join(classfolder, image))
                    train_x.append(resize_image(img))
                    train_y.append(file)

        train_x = np.array(train_x)
        train_y = LabelEncoder().fit_transform(train_y)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)
        return (train_x, train_y), (test_x, test_y)

    def __init__(self, path):
        self.path = os.path.join(path, 'flowers/flower_photos')
        (train_x, train_y),(test_x, test_y)  = self.load_data()
        train_x, test_x = resize_images(train_x), resize_images(test_x)
        train_y = utils.to_categorical(train_y, self.num_classes)
        test_y = utils.to_categorical(test_y, self.num_classes)
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y

    @property
    def num_classes(self):
        return 5
