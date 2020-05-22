import os
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from termcolor import colored
from common.callbacks import StopTraining
from common.image_generators import ImageDataFromDisk
from common.model import Model

class BinaryImageClassifier(Model):
    ''' A CNN based binary image classifier. '''

    def __init__(self, path):
        '''
        Initialize and compile model.
        Args:
            path: top level location of images on disk with proper folder structure
        '''

        Model.__init__(self)
        self.image_size = (150, 150)  # pixels

        # set path to model data
        if os.path.isdir(path):
            self.path = path if path.endswith('/') else path + '/'
        else:
            print(colored('error: "{}" not found', 'magenta').format(path))
            exit()

        # define model
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=keras.activations.relu,
                input_shape=self.image_size + (3,)
            ),
            keras.layers.MaxPool2D(
                pool_size=(2, 2)
            ),
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation=keras.activations.relu
            ),
            keras.layers.MaxPool2D(
                pool_size=(2, 2)
            ),
            keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation=keras.activations.relu
            ),
            keras.layers.MaxPool2D(
                pool_size=(2, 2)
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                units=512,
                activation=keras.activations.relu
            ),
            keras.layers.Dense(
                units=1,
                activation=keras.activations.sigmoid
            )
        ])

        # define training metrics
        self.metrics = [
            keras.metrics.BinaryAccuracy(name='acc'),
            keras.metrics.AUC(name='auc')
        ]

        # compile model
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=self.metrics
        )

    def train(self):
        ''' Train model using predefined train and test image generators. '''

        # check if train and test folders are present (relative to set path)
        dirs = ['train', 'test']
        self.check_dirs(dirs)

        # define train image generator with on the fly image augmentation
        train_data = ImageDataFromDisk(
            path=self.path + dirs[0],
            target_size=self.image_size,
            class_mode='binary',
            batch_size=100,
            augment=True
        )

        # define test image generator
        test_data = ImageDataFromDisk(
            path=self.path + dirs[1],
            target_size=self.image_size,
            class_mode='binary',
            batch_size=100,
            augment=False
        )

        # train model
        callbacks = [StopTraining('acc', 0.95)]
        history = self.model.fit(
            train_data.image_data_generator(),
            epochs=5,
            validation_data=test_data.image_data_generator(),
            callbacks=callbacks
        )

        # update train history
        self.history = self.history.append(pd.DataFrame(history.history), ignore_index=True)

    def predict(self):
        ''' Predict image classes for all images in "../predict" folder. '''

        # check if train and predict folders are present (relative to set path)
        dirs = ['train', '../predict']
        self.check_dirs(dirs)

        # read image classes
        train_classes = os.listdir(self.path + dirs[0])

        # loop over found images
        predict_path = self.path + dirs[1]
        for img_name in os.listdir(predict_path):

            # process image
            img = keras.preprocessing.image.load_img(
                predict_path + '/' + img_name,
                target_size=self.image_size
            )
            img_arr = keras.preprocessing.image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)

            # predict
            prediction = self.model(img_arr)
            class_name = train_classes[int(prediction.numpy().flatten()[0])]
            print('{} --> {}'.format(img_name, class_name))
