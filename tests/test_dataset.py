import os
import sys
import tensorflow as tf
import pytest
import configparser
import math

#Get the parameters
config = configparser.ConfigParser()
config.read('tests/test_parameters.ini')

img_size = config.getint('setting', 'img_size')
batch = config.getint('setting', 'batch')
epochs = config.getint('setting', 'epochs')
classes = config.get('setting', 'classes').split(',')
num_classes = config.getint('setting', 'num_classes')
train_params = {
    'label_mode': config.get('setting', 'label_mode'),
    'color_mode': config.get('setting', 'color_mode'),
    'batch_size': config.getint('setting', 'batch'),
    'image_size': eval(config.get('setting', 'image_size')),
    'seed': config.getint('setting', 'seed')}

# Get the current file's absolute path and move backward to 
# get the folders' paths and add the directory to the Python module search path
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)

img_path = os.path.join(grandparent_dir, config.get('path', 'img_path'))
test_img_path = os.path.join(grandparent_dir, config.get('path', 'test_img_path'))
weight_path = os.path.join(grandparent_dir, config.get('path', 'weight_path'))

os.chdir(grandparent_dir)
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from classificationmodel.dataset import dataset_generator

@pytest.fixture(scope="module")
def data_generators():
    """
    This fixture prepares the datasets to use in the tests.
    GIVEN:
        - The dataset generator function from dataset.py is called
          with the parameters defined in test_parameters.ini.
    WHEN:
        - dataset_generator builds the datasets.
    THEN:
        - Training, validation, and test set generators are created 
          using the dataset generator function.
        - The generators are returned for use in the tests.
    """
    train_set, val_set, test_set = dataset_generator(img_path, 
                                                     test_img_path, 
                                                     train_params)
    return train_set, val_set, test_set

def test_dataset_generator_returns_tuple(data_generators):
    """
    This test checks if the dataset_generator function returns three objects
    of type tf.data.Dataset, which are used for training, validation
    and testing the model respectively.
    GIVEN:
        - The data_generators fixture function.
    WHEN:
        - The dataset generator function returns three datasets.
    THEN:
        - The returned datasets correspond to three objects of 
          type tf.data.Dataset.
    """
    train_set, val_set, test_set = data_generators
    assert isinstance(train_set, tf.data.Dataset)
    assert isinstance(val_set, tf.data.Dataset)
    assert isinstance(test_set, tf.data.Dataset)

def test_dataset_generator_length(data_generators):
    """
    This test checks if the length of the training,
    validation and test sets generated are greater than zero,
    and that the number of batches is the expected one.
    GIVEN:
        - The data_generators fixture function.
    WHEN:
        - The dataset generator function returns three datasets.
        - The expected number of batches for all of the datasets are 
          calculated rounding to the next integer, 
          knowing the number of images and of the batches.
    THEN:
        - The number of batches in each dataset is greater than zero.
        - The number of batches per each dataset matches the expected 
          value based on the batch size and number of images.
        - The number of batches in the validation and test sets matches
          each other due to the same number of images in each set.
    """
    train_set, val_set, test_set = data_generators
    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(test_set) > 0
    #900 is the expected number of images in the train set
    expected_train_batches = math.ceil(900 / batch)
    assert len(train_set) == expected_train_batches
    #100 is the expected number of images in both the val and test sets
    expected_val_batches = math.ceil(100 / batch) 
    assert len(val_set) == len(test_set)
    assert len(val_set) == expected_val_batches

def test_dataset_generator_image_shape_dtype(data_generators):
    """
    This test checks if the shape and dtype of the images and labels
    in the datasets match the expected values.
    GIVEN:
        - The data_generators fixture function.
    WHEN:
        - The dataset generator function returns three datasets.
        - The batches of the train_set are accessed by transforming the dataset
          into an terable object.
    THEN:
        - The tensor shape matches the given one through the parameters.
    """
    train_set, _, _ = data_generators
    x_train, y_train = next(iter(train_set))
    assert x_train.shape == tf.TensorShape([train_params['batch_size'],
                                            train_params['image_size'][0], 
                                            train_params['image_size'][1],
                                            3])
    assert x_train.dtype == 'float32'
    assert y_train.shape == (train_params['batch_size'], num_classes)
    assert y_train.dtype == 'float32'


def test_dataset_generator():
    """
    This test checks if the images are loaded from a given directory.
    GIVEN:
        - A test image folder with 10 images of size 3x3 pixels.
        - Parameters to create the datasets.
    WHEN:
        - The dataset_generator function is executed in this folder.
    THEN:
        - Training, validation, and test sets are generated
          from the given folder.
        - The length of the training set matches the 90% of 
          images in the test folder.
        - The length of the validation set matches the 10% of
          images in the test folder.
        - The length of the test set is the same as the number
          of images in the test folder.
    """
    test_train_params = {
        "label_mode": "categorical",
        "color_mode": "rgb",
        "batch_size": 1,
        "image_size": (3, 3),
        "seed": 42,
    }
    train_set_test, val_set_test, test_set_test = dataset_generator(
                                                    'tests/test_dataset_images', 
                                                    'tests/test_dataset_images',
                                                    test_train_params)
    assert len(train_set_test) == 9
    assert len(val_set_test) == 1
    assert len(test_set_test) == 10