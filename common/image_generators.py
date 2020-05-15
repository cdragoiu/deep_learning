import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageDataFromDisk:
    ''' Set up an image data pipeline based on disk files. '''

    def __init__(self, path, target_size, class_mode, batch_size, augment):
        '''
        Args:
            path:        top level location of images on disk
                         (proper directory structure must be present)
            target_size: final image resolution (height, width)
            batch_size:  how many images will be process per batch
            class_mode:  one of ImageDataGenerator class modes
            augment:     true to augment data, false otherwise
        '''

        self.path = os.path.abspath(path)
        self.target_size = target_size
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.augment = augment

    def image_data_generator(self):
        ''' Returns a directory iterator that generates pairs of images and labels. '''

        if self.augment:
            generator = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode='nearest',
                horizontal_flip=True,
                rescale=1/255
            )
        else:
            generator = ImageDataGenerator(rescale=1/255)
        return generator.flow_from_directory(
            self.path,
            target_size=self.target_size,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
        )
