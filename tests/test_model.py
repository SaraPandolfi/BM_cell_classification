import os
import sys
import pytest
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.applications.efficientnet import EfficientNetB3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import configparser
import json
import filecmp
import numpy as np
import math
import json
# Specify the absolute file path of the configuration file
config_file_path = 'tests/test_parameters.json'
config = configparser.ConfigParser()
with open(config_file_path) as config_file:
    config = json.load(config_file)

batch = config['setting']['batch']
epochs = config['setting']['epochs']
classes = config['setting']['classes']
num_classes = config['setting']['num_classes']
train_params = config['setting']['train_params']

# Get the current file's absolute path and move backward to 
# get the folders' paths and add the directory to the Python module search path
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)

img_path = os.path.join(grandparent_dir, config['path']['img_path'])
test_img_path = os.path.join(grandparent_dir, config['path']['test_img_path'])
weight_path = os.path.join(grandparent_dir, config['path']['weight_path'])
model_path = os.path.join(grandparent_dir, model_path = config['path']['model_path'])

os.chdir(grandparent_dir)
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from classificationmodel.dataset import dataset_generator
from classificationmodel.model import (build_model, 
                                       load_model_weights, 
                                       train_model, 
                                       load_model,
                                       save_model)

train_set, val_set, _ = dataset_generator(img_path,
                                          test_img_path,
                                          train_params)

@pytest.fixture(scope="module")
def efficientnet():
    """
    This fixture function loads a pre-trained EfficientNet model
    from the 'model.json' file stored at the model_path
    or builds it if the file is not found.
    GIVEN:
        - The 'model.json' file is present or not.
    WHEN:
        - The EfficientNet model is loaded from the file if present.
    THEN:
        - If the file is not found, the model is built.
        - The model is returned.
    """
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        model = build_model(num_classes)
    except Exception as e:
        pytest.fail(f"Failed to load or build model: {e}")
    return model

def test_build_model(efficientnet):
    """
    This test checks if the build_model function returns an instance
    of the tf.keras.Model class with the expected output shape.
    GIVEN:
        - A efficientnet model.
        - A number of classes for classification.
    WHEN:
        - The build_model function is called with the number of classes.
    THEN:
        - The returned model is an instance of tf.keras.Model.
        - The output shape of the model matches the expected shape.
    """
    assert isinstance(efficientnet, tf.keras.Model)
    assert efficientnet.output_shape == (None, num_classes)

def test_load_model_weights_existing_file(efficientnet):
    """
    This test checks the load_model_weights function when
    the weights file exists.
    GIVEN:
        - A EfficientNet model.
        - The path to an existing weights file.
    WHEN:
        - The load_model_weights function is called with the model
          and weights file path.
    THEN:
        - The loaded model is still an efficientnet model.
        - The file exists.
        - The weights of the efficientnet and of the loaded 
          model are different.
    """
    # Get the original model's weights
    original_weights = efficientnet.get_weights()
    loaded_model = load_model_weights(efficientnet, weight_path)
    loaded_weights = loaded_model.get_weights()
    assert loaded_model == efficientnet
    assert os.path.exists(weight_path)
    assert not np.array_equal(loaded_weights, original_weights)

def test_load_model_weights_nonexistent_file(efficientnet):
    """
    Test to verify the load_model_weights function when 
    the weights file does not exist.
    GIVEN:
        - A EfficientNet model.
        - The path to a non-existent weights file.
    WHEN:
        - The load_model_weights function is called with the model
          and non-existent weights file path.
    THEN:
        - The function returns an efficient model.
        - The file doesn't exist.
        - The length of the weights arrays is the same.
        - The weights are the same.
    """
    # Get the original model's weights
    original_weights = efficientnet.get_weights()

    # Call the load_model_weights function with a non-existent file path
    loaded_model = load_model_weights(efficientnet, 'nonexistent_weights.h5')

    assert loaded_model == efficientnet
    loaded_weights = loaded_model.get_weights()
    assert not os.path.exists('nonexistent_weights.h5')
    assert len(loaded_weights) == len(original_weights)
    for loaded_layer_weights, original_layer_weights in zip(loaded_weights, original_weights):
        assert np.array_equal(loaded_layer_weights, original_layer_weights)

def test_compile_model(efficientnet):
    """
    This test checks if the model returned by the build_model function
    has been compiled with the expected optimizer and loss function.
    GIVEN:
        - A EfficientNet model.
    WHEN:
        - The compile_model function is called with the model.
    THEN:
        - The optimizer of the model is an instance of Adam.
        - The loss function of the model is 'categorical_crossentropy'.
    """
    assert efficientnet.optimizer.__class__ == Adam
    assert efficientnet.loss.__name__ == 'categorical_crossentropy'

def test_save_and_load_model(efficientnet):
    """
    This test verifies the functionality of saving and loading a model.
    GIVEN:
        - A EfficientNet model.
    WHEN:
        - The save_model and load_model functions are used to
          save and load the model.
    THEN:
        - The saved and loaded models are compared to ensure they are identical.   
    """
    original_model_path = 'original_model.json'
    save_model(efficientnet, original_model_path)
    # Load the saved model
    loaded_model = load_model(original_model_path)    
    # Save the loaded model
    loaded_model_path = 'loaded_model.json'
    save_model(loaded_model, loaded_model_path)
    
    # Compare the contents of the two saved files
    assert filecmp.cmp(original_model_path, loaded_model_path), (
        "Saved models are not identical")
    
def test_train_model_returns_history(efficientnet):
    """
    This test checks if the train_model function returns a history object
    with the expected keys, indicating that the model has been trained.
    GIVEN:
        - A EfficientNet model.
        - Training and validation datasets.
        - Batch size and number of epochs.
        - A path to save the weights.
    WHEN:
        - The train_model function is called with the given parameters.
    THEN:
        - The returned history object has the expected keys.
    """
    history, _ = train_model(efficientnet, 
                             train_set, 
                             val_set, 
                             batch, 
                             epochs, 
                             weight_path)
    assert isinstance(history.history, dict)
    assert 'accuracy' in history.history.keys()
    assert 'val_accuracy' in history.history.keys()
    assert 'loss' in history.history.keys()
    assert 'val_loss' in history.history.keys()

def test_model_callbacks_attribute(efficientnet):
    """
    This test checks if the trained model has a 'callbacks'
    attribute after training.
    GIVEN:
        - A EfficientNet model.
        - Training and validation datasets.
        - Batch size and number of epochs.
        - A path to save the weights.
    WHEN:
        - The train_model function is called with the given parameters.
    THEN:
        - The trained model has a 'callbacks' attribute.
    """
    _, trained_model = train_model(efficientnet,
                                   train_set,
                                   val_set,
                                   batch,
                                   epochs,
                                   weight_path)
    assert hasattr(trained_model, 'callbacks') == True, (
    "Model does not have 'callbacks' attribute after training")

def test_model_callbacks_instance(efficientnet):
    """
    This test checks if the callbacks attribute of the trained model
    contains instances of the ReduceLROnPlateau and ModelCheckpoint classes.
    GIVEN:
        - A EfficientNet model.
        - Training and validation datasets.
        - Batch size and number of epochs.
        - A path to save the weights.
    WHEN:
        - The train_model function is called with the given parameters.
    THEN:
        - The 'callbacks' attribute of the trained model contains instances
          of the ReduceLROnPlateau and ModelCheckpoint classes.
    """
    _, trained_model = train_model(efficientnet, 
                                   train_set, 
                                   val_set, 
                                   batch, 
                                   epochs,
                                   weight_path)
    expected_callbacks = (ModelCheckpoint, ReduceLROnPlateau)
    actual_callbacks = trained_model.callbacks
    assert isinstance(actual_callbacks[0], expected_callbacks[0]), (
        "First callback is not an instance of ModelCheckpoint")
    assert isinstance(actual_callbacks[1], expected_callbacks[1]), (
        "Second callback is not an instance of ReduceLROnPlateau")  