import math, os
from utilities import get_checkpoint_file, copy_and_overwrite, delete_file
from keras import models, layers, optimizers
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from abc import ABC, abstractmethod
import numpy as np
from resnet import resnet
from ENAS_Keras.ENAS import EfficientNeuralArchitectureSearch
from ENAS_Keras.src.utils import sgdr_learning_rate


def find_highest_acc(history):
    max_index, max_acc = None, 0
    for index, acc in enumerate(history.history['val_acc']):
        if acc > max_acc:
            max_acc = acc
            max_index = index
    return max_index + 1, max_acc


def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


class Trainer(ABC):
    def __init__(self, dataset=None):
        self.set_dataset(dataset) if dataset is not None else None

    @abstractmethod
    def build(self):
        #should return the compiled model
        pass

    def set_dataset(self, dataset):
        self._dataset = dataset
        self._model = self.build()

    @property
    def checkpoint_file(self):
        return get_checkpoint_file(self._dataset.name, type(self).__name__)

    @property
    def checkpoint_directory(self):
        return get_checkpoint_file(self._dataset.name, type(self).__name__, is_dir=True)

    @property
    def val_acc(self):
        if self._val_acc is None:
            raise Exception('Model should be evaluated using evaluate() before accessing val_acc')
        return self._val_acc

    @property
    def best_model_path(self):
        if self._best_epoch_num is None:
            raise Exception('Model should be evaluated using evaluate() before accessing best_model_path')
        return self.checkpoint_file.format(epoch=self._best_epoch_num, val_acc=self.val_acc)

    def evaluate(self, epochs, batch_size):
        if self._dataset is None:
            raise Exception('dataset should be set using set_dataset()')
        callbacks = [
                        ModelCheckpoint(filepath=self.checkpoint_file,
                                        monitor='val_acc',
                                        save_best_only=True),
                        LearningRateScheduler(lr_schedule),
                        ReduceLROnPlateau(factor=np.sqrt(0.1),
                                          cooldown=0,
                                          patience=5,
                                          min_lr=0.5e-6),
                        EarlyStopping(monitor='val_acc', patience=3)
                    ]
        history = self._model.fit(self._dataset.train_x, self._dataset.train_y,
                                  epochs=epochs, batch_size=batch_size,
                                  validation_data=(self._dataset.test_x, self._dataset.test_y),
                                  shuffle=True, callbacks=callbacks)
        self._best_epoch_num, self._val_acc = find_highest_acc(history)


class ProbenetTrainer(Trainer):
    def build(self):
        r = math.floor(math.log10(self._dataset.num_classes))
        fc_units = (512 + self._dataset.num_classes) // 2
        num_filters = [8, 16, 32, 64, 128]

        inputs = layers.Input(shape=self._dataset.instance_shape)
        model = inputs
        for filters in num_filters:
            for i in range(r):
                model = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same') (model)
                model = layers.BatchNormalization() (model)
                model = layers.Activation('relu') (model)
            model = layers.MaxPool2D(pool_size=2, strides=2) (model)
        model = layers.Flatten() (model)
        model = layers.Dense(units=fc_units, activation='relu') (model)
        model = layers.Dense(units=self._dataset.num_classes, activation='softmax') (model)
        probenet = models.Model(inputs=inputs, outputs=model)

        optimizer = optimizers.Adam(lr=lr_schedule(0))
        probenet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return probenet


class ResnetTrainer(Trainer):
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    def build(self):
        n = 3
        depth = n * 6 + 2
        model = resnet(input_shape=self._dataset.instance_shape, depth=depth,
                       num_classes=self._dataset.num_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        return model


class TransferLearner(Trainer):
    def __init__(self, dataset=None, model=None):
        if dataset is not None:
            self.set_dataset(dataset)
            if model is not None:
                self.transfer_from_model(model)
        if model is not None:
            raise Exception('dataset should be specified')

    def set_dataset(self, dataset):
        self._dataset = dataset

    def transfer_from_model(self, model_path):
        model = load_model(model_path)
        self._model = self.build(model)

    @abstractmethod
    def build(self, model):
        pass


class StandardTransferLearner(TransferLearner):
    def build(self, model):
        # TODO: Freeze layers
        model.layers.pop()
        output = layers.Dense(self._dataset.num_classes,
                              activation='softmax',
                              kernel_initializer='he_normal')(model.layers[-1].output)
        model = models.Model(inputs=model.input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        return model


class EnasTrainer(Trainer):
    @property
    def best_model_path(self):
        return self.checkpoint_directory

    def build(self):
        nt = sgdr_learning_rate(n_Max=0.05, n_min=0.001, ranges=2, init_cycle=4)
        model = EfficientNeuralArchitectureSearch(
            x_train=self._dataset.train_x,
            y_train=self._dataset.train_y,
            x_test=self._dataset.test_x,
            y_test=self._dataset.test_y,
            search_epochs=len(nt),
            child_lr_scedule=nt,
            child_network_name=self._dataset.name,
            child_classes=self._dataset.num_classes,
            child_input_shape=self._dataset.instance_shape,
            working_directory=self.checkpoint_directory,
            child_init_filters=32,
            num_nodes=4,
            num_opers=3,
            run_on_jupyter=False,
            initialize_child_weight_directory=False,
            save_to_disk=True,
        )

        return model

    def evaluate(self, epochs, batch_size):
        if self._dataset is None:
            raise Exception('dataset should be set using set_dataset()')

        # number of samples of child networks is found from nt
        # i.e., epochs != number of search epochs of ENAS (number of child networks trained)
        self._model.controller_epochs = epochs
        self._model.child_epochs = epochs
        self._model.child_batch_size = batch_size
        self._model.child_val_batch_size = batch_size,
        #self._model.data_gen = self._dataset.get_data_gen(),
        #self._model.data_flow_gen = self._dataset.get_data_flow_gen(batch_size)


        self._model.search_neural_architecture()

        print(self._model.best_normal_cell)
        print(self._model.best_reduction_cell)

        self._model.train_best_cells(child_epochs=epochs * 2)

        self._best_epoch_num, self._val_acc = self._model.best_epoch_num, self._model.best_val_acc


class EnasTransferLearner(EnasTrainer, TransferLearner):
    def transfer_from_model(self, model_path):
        copy_and_overwrite(model_path, self.checkpoint_directory)
        delete_file(os.path.join(self.checkpoint_directory, '{}_record.csv'.format(self._dataset.name)))
        self._model = self.build()

    def evaluate(self, epochs, batch_size):
        EnasTrainer.evaluate(self, epochs, batch_size)
