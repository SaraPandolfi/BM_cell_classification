import tensorflow as tf
from dataset import dataset_generator
from parameters import BATCH, IMG_SIZE, classes, img_path, augmentation_params, train_params
from keras.preprocessing.image import ImageDataGenerator
import pytest

@pytest.fixture(scope="module")
def train_val_generators():
    """
    This fixture function creates training and validation set generators using the
    dataset_generator function from dataset.py with the parameters defined in parameters.py.
    It returns the generators for use in the tests.
    """
    train_set, val_set = dataset_generator(img_path, augmentation_params, train_params)
    return train_set, val_set

def test_dataset_generator_returns_tuple(train_val_generators):
    """
    This test checks if the dataset_generator function returns two objects of type DirectoryIterator,
    which are used for splitting the total dataset in the training and validation data respectively.
    """
    train_set, val_set = train_val_generators
    assert isinstance(train_set, tf.keras.preprocessing.image.DirectoryIterator)
    assert isinstance(val_set, tf.keras.preprocessing.image.DirectoryIterator)

def test_dataset_generator_length(train_val_generators):
    """
    This test checks if the length of the training and validation sets generated are greater than zero.
    """
    train_set, val_set = train_val_generators
    assert len(train_set) > 0
    assert len(val_set) > 0

def test_dataset_generator_image_shape_dtype(train_val_generators):
    """
    This test checks if the shape and dtype of the images and labels generated match the expected values.
    """
    train_set, val_set = train_val_generators
    x_train, y_train = train_set.next()
    assert x_train.shape == (BATCH, IMG_SIZE, IMG_SIZE, 3)
    assert x_train.dtype == 'float32'
    assert y_train.shape == (BATCH, len(classes))
    assert y_train.dtype == 'float32'

def test_dataset_generator_augmentation(train_val_generators):
    """
    This test function checks if the dataset_generator function performs data augmentation
    by comparing two batches of training data to check if they are different.
    """
    train_set, val_set = train_val_generators
    x_train_orig, y_train_orig = train_set.next()
    x_train_aug, y_train_aug = train_set.next()
    assert not tf.reduce_all(tf.equal(x_train_orig, x_train_aug))
    assert not tf.reduce_all(tf.equal(y_train_orig, y_train_aug))

def test_dataset_generator_params(train_val_generators):
    """
    This test checks if the parameters of the training and validation sets generated match the expected values.
    """
    train_set, val_set = train_val_generators
    assert train_set.batch_size == BATCH
    assert train_set.image_shape == (IMG_SIZE, IMG_SIZE, 3)
    assert len(train_set.class_indices) == len(classes)
