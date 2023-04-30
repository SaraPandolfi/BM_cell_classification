import os
import pytest
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.applications.efficientnet import EfficientNetB3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from parameters import BATCH, EPOCHS, num_classes
from dataset import train_set, val_set
from model import build_model, train_model, load_model


@pytest.fixture(scope="module")
def efficientnet():
    """
    This fixture function loads a pre-trained EfficientNet model from a pickled file
    and makes it available to the test functions.
    """
    try:
        model = load_model('model.pkl')
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
    return model


def test_build_model(efficientnet):
    """
    This test checks if the build_model function returns
    an instance of the tf.keras.Model class with the expected output shape.
    """
    assert isinstance(efficientnet, tf.keras.Model)
    assert efficientnet.output_shape == (None, num_classes)

def test_compile_model(efficientnet):
    """
    This test checks if the model returned by the build_model function
    has been compiled with the expected optimizer, loss function, and accuracy for metrics.
    """
    assert efficientnet.optimizer.__class__ == Adam
    assert efficientnet.loss.__name__ == 'categorical_crossentropy'
    assert 'accuracy' in efficientnet.metrics_names

def test_train_model_returns_history(efficientnet):
    """
    This test checks if the train_model function returns a history object
    with the expected keys, indicating that the model has been trained.
    """
    history, trained_model = train_model(efficientnet, train_set, val_set, BATCH, EPOCHS)
    assert isinstance(history.history, dict)
    assert 'accuracy' in history.history.keys()
    assert 'val_accuracy' in history.history.keys()
    assert 'loss' in history.history.keys()
    assert 'val_loss' in history.history.keys()


def test_model_callbacks_attribute():
    """
    This test checks if the trained model has a callbacks attribute after training.
    """
    efficientNet = build_model(num_classes)
    model_history, trained_model = train_model(efficientNet, train_set, val_set, BATCH, EPOCHS)
    assert hasattr(trained_model, 'callbacks') == True, "Model does not have 'callbacks' attribute after training"


def test_model_callbacks_instance():
    """
    This test checks if the callbacks attribute of the trained model
    contains instances of the ReduceLROnPlateau and ModelCheckpoint classes.
    """
    efficientNet = build_model(num_classes)
    model_history, trained_model = train_model(efficientNet, train_set, val_set, BATCH, EPOCHS)
    assert all(isinstance(callback, (ReduceLROnPlateau, ModelCheckpoint)) for callback in trained_model.callbacks) == True, "Model callbacks are not instances of ReduceLROnPlateau and ModelCheckpoint classes"



