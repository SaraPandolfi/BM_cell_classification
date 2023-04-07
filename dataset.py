import tensorflow as tf
from tensorflow.python.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from parameters import BATCH, IMG_SIZE, classes, img_path, augmentation_params, train_params


def dataset_generator(img_path, augmentation_params, train_params):
    '''
    Takes the path to a directory and generates batches of data
    Create DataGenerator yielding tuples of (x, y) with shape (batch_size, height, width, channels) 
    where x is the input image and y is the ground-truth.
    The data generation and its split are performed using augmentation_params and train_params:
    augmentation_params give information regarding the possible flipping of the images and the test/train set split ration
    train_params give information regarding the image dimensions, color mode, batch size, classes, shuffle and seed
    Parameters
    ----------
    img_path : str
        path for the images directory
    augmentation_params : dict
        dict of keras ImageDataGenerator args for the generation of custom images;
        traning and validation datasets split ratio
    train_params : dict
        size of the images;
        size of the batches of data;
        seed for randomness control.
    Returns
    -------
    Tf DirectoryIterator yielding tuples of (x, y) where:
    x is a numpy array containing a batch of images with shape (batch_size, height, width, channels) 
    y is a numpy array of corresponding labels

    '''

    img_data_gen = ImageDataGenerator(**augmentation_params, rescale=1./255)

    train_img_generator = img_data_gen.flow_from_directory(img_path,
                                                           **train_params, 
                                                           subset='training')
    
    val_img_generator = img_data_gen.flow_from_directory(img_path, 
                                                         **train_params, 
                                                         subset='validation')
    

    return train_img_generator, val_img_generator

train_set, val_set = dataset_generator(img_path, augmentation_params, train_params)
