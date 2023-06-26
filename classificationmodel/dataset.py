import tensorflow as tf
from keras.utils import image_dataset_from_directory
from parameters import img_path, test_img_path, train_params


def dataset_generator(img_path, test_img_path, train_params):
    """
    Takes the path to a directory and generates batches of data.
    
    Create a tf.data.Dataset yielding tuples of (x, y) with shape 
    (batch_size, height, width, channels) where x is the input image
    and y is the ground-truth.
    
    The data generation and its split are performed using train_params.
    They give information regarding the image dimensions, color mode,
    batch size, classes, shuffle, and seed.
    
    Parameters:
        img_path (str): Path for the training and validation images directory.
        test_img_path (str): Path for the test images directory.
        train_params (dict): Dictionary containing labels and images color mode,
            size of the batches of data, size of the images, and seed for
            randomness control.
    
    Returns:
        A tf.data.Dataset object yielding tuples of (x, y) where:
        - x is a numpy array containing a batch of images with shape
          (batch_size, height, width, channels).
        - y is a numpy array of corresponding labels.
    """
    train_img_generator = image_dataset_from_directory(img_path,    
                                                       shuffle=True,
                                                       validation_split=0.10,  
                                                       subset='training',
                                                       **train_params)
    
    val_img_generator = image_dataset_from_directory(img_path,
                                                     shuffle=True,
                                                     validation_split=0.10,
                                                     subset='validation',
                                                     **train_params)
    
    test_dataset = image_dataset_from_directory(test_img_path,
                                                shuffle=False,
                                                **train_params)

    return train_img_generator, val_img_generator, test_dataset


train_set, val_set, test_set = dataset_generator(img_path,
                                                 test_img_path,
                                                 train_params)