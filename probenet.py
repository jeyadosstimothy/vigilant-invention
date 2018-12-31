import math
from keras import models, layers, optimizers
from utilities import get_checkpoint_file
from keras.callbacks import ModelCheckpoint

class Prober:
    def __init__(self, dataset, lr=0.01):
        self.dataset = dataset
        self.probenet = self.build(lr=lr)


    def build(self, lr):
        r = math.floor(math.log10(self.dataset.num_classes))
        fc_units = (512 + self.dataset.num_classes) // 2
        num_filters = [8, 16, 32, 64, 128]

        inputs = layers.Input(shape=self.dataset.instance_shape)
        model = inputs
        for filters in num_filters:
            for i in range(r):
                model = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same') (model)
                model = layers.BatchNormalization() (model)
                model = layers.Activation('relu') (model)
            model = layers.MaxPool2D(pool_size=2, strides=2) (model)
        model = layers.Flatten() (model)
        model = layers.Dense(units=fc_units, activation='relu') (model)
        model = layers.Dense(units=self.dataset.num_classes, activation='softmax') (model)
        probenet = models.Model(inputs=inputs, outputs=model)

        optimizer = optimizers.Adam(lr=lr)
        probenet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return probenet


    def evaluate(self, epochs=1):
        checkpoint_file = get_checkpoint_file(self.dataset.name)
        callbacks = [
                        ModelCheckpoint(checkpoint_file, monitor='val_acc', save_best_only=True),
                    ]
        history = self.probenet.fit(self.dataset.train_x, self.dataset.train_y,
                                    validation_data=(self.dataset.test_x, self.dataset.test_y),
                                    epochs=epochs, callbacks=callbacks)
        return max(history.history['val_acc'])
