import os
import sys
import tensorflow as tf
import pytest
import configparser

#Get the parameters
config = configparser.ConfigParser()
config.read('test_parameters.ini')

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
    This fixture function creates training, validation and test set generators
    using the dataset_generator function from dataset.py with the parameters
    defined in parameters.py.
    It returns the generators for use in the tests.
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
    """
    train_set, val_set, test_set = data_generators
    assert isinstance(train_set, tf.data.Dataset)
    assert isinstance(val_set, tf.data.Dataset)
    assert isinstance(test_set, tf.data.Dataset)

def test_dataset_generator_length(data_generators):
    """
    This test checks if the length of the training,
    validation and test sets generated are greater than zero.
    """
    train_set, val_set, test_set = data_generators
    assert tf.data.experimental.cardinality(train_set) > 0
    assert tf.data.experimental.cardinality(val_set) > 0
    assert tf.data.experimental.cardinality(test_set) > 0

def test_dataset_generator_image_shape_dtype(data_generators):
    """
    This test checks if the shape and dtype of the images and labels generated
    match the expected values.
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