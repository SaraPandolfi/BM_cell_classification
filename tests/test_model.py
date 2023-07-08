import os
import sys
import pytest
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.applications.efficientnet import EfficientNetB3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Get the current file's absolute path and move backward to 
# add the directory to the Python module search path
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
os.chdir(grandparent_dir)
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from classificationmodel.dataset import dataset_generator
from classificationmodel.model import (build_model, 
                                       load_model_weights, 
                                       train_model, 
                                       load_model)
from classificationmodel.parameters import (img_path,
                                            test_img_path, 
                                            train_params, 
                                            batch, 
                                            epochs, 
                                            num_classes,
                                            weight_path)

train_set, val_set, _ = dataset_generator(img_path,
                                          test_img_path,
                                          train_params)

@pytest.fixture(scope="module")
def efficientnet():
    """
    This fixture function loads a pre-trained EfficientNet model
    from the 'model.pkl' file or builds it if the file is not found.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        model = build_model(num_classes)
    except Exception as e:
        pytest.fail(f"Failed to load or build model: {e}")
    return model

def test_build_model(efficientnet):
    """
    This test checks if the build_model function returns
    an instance of the tf.keras.Model class with the expected output shape.
    """
    assert isinstance(efficientnet, tf.keras.Model)
    assert efficientnet.output_shape == (None, num_classes)

def test_load_model_weights_existing_file(efficientnet):
    """
    This test checks the load_model_weights function when the weights file exists.
    It verifies that the function successfully loads the weights into the given model.
    """
    loaded_model = load_model_weights(efficientnet, str(weight_path))
    assert loaded_model == efficientnet

def test_load_model_weights_nonexistent_file(efficientnet):
    """
    This test checks the load_model_weights function when the weights
    file does not exist by calling the function with a non-existent file.
    It verifies that the function returns False and does not load any weights.
    """
    loaded_model = load_model_weights(efficientnet, 'nonexistent_weights.h5')
    assert loaded_model == False

def test_compile_model(efficientnet):
    """
    This test checks if the model returned by the build_model function
    has been compiled with the expected optimizer and loss function.
    """
    assert efficientnet.optimizer.__class__ == Adam
    assert efficientnet.loss.__name__ == 'categorical_crossentropy'

def test_train_model_returns_history(efficientnet):
    """
    This test checks if the train_model function returns a history object
    with the expected keys, indicating that the model has been trained.
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

def test_model_callbacks_attribute():
    """
    This test checks if the trained model has 
    a callbacks attribute after training.
    """
    efficientNet = build_model(num_classes)
    _, trained_model = train_model(efficientNet,
                                   train_set,
                                   val_set,
                                   batch,
                                   epochs,
                                   weight_path)
    assert hasattr(trained_model, 'callbacks') == True, (
    "Model does not have 'callbacks' attribute after training")

def test_model_callbacks_instance():
    """
    This test checks if the callbacks attribute of the trained model
    contains instances of the ReduceLROnPlateau and ModelCheckpoint classes.
    """
    efficientNet = build_model(num_classes)
    _, trained_model = train_model(efficientNet, 
                                   train_set, 
                                   val_set, 
                                   batch, 
                                   epochs,
                                   weight_path)
    expected_callbacks = [ReduceLROnPlateau, ModelCheckpoint]    
    actual_callbacks = trained_model.callbacks
    
    assert all(isinstance(callback, tuple(expected_callbacks))
               for callback in actual_callbacks), (
        "Model callbacks are not instances of ReduceLROnPlateau and ModelCheckpoint classes")