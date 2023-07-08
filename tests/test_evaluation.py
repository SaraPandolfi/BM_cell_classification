import os
import sys
import pytest
import numpy as np
import tensorflow as tf
import math
import configparser

#Get the parameters
config = configparser.ConfigParser()
config.read('test_parameters.ini')
img_size = config.getint('setting', 'img_size')
batch = config.getint('setting', 'batch')
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

from classificationmodel.model import load_model, build_model
from classificationmodel.dataset import dataset_generator
from classificationmodel.evaluation import evaluate_model, evaluation_report

_, _, test_set = dataset_generator(img_path,
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

def test_evaluate_model(efficientnet):
    """
    This verifies that the loss and accuracy are within a certain tolerance
    close to the actual values.
    """  
    test_loss, test_accuracy = evaluate_model(efficientnet, 
                                              test_set)
    expected_loss = 1.10  
    expected_accuracy = 0.8  
    tolerance = 1e-1 
    assert (np.abs(test_loss - expected_loss) < tolerance,
            f"Test Loss does not match expected value. "
            f"Expected: {expected_loss}, Actual: {test_loss}")
    assert (np.abs(test_accuracy - expected_accuracy) < tolerance,
            f"Test Accuracy does not match expected value. "
            f"Expected: {expected_accuracy}, Actual: {test_accuracy}")

def test_evaluation_report_lists(efficientnet):
    y_true = []
    y_pred = []
    for x, y in test_set:
        assert (np.array_equal(x[0], y[0]),
                f"Mismatch between x and y in the test_dataset")
        
        y_pred_probs = efficientnet.predict(x)
        # Check if the shape of y and y_pred_prob are the same
        #  so they can be treated equally
        assert y.shape == y_pred_probs.shape
        assert y.shape[1] == num_classes, "Incorrect shape of 'y' array"
        assert y_pred_probs.shape[1] == num_classes, "Incorrect shape of predicted probabilities"  
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(y_pred_probs, axis=1))

   # Compare the lengths of y_true and y_pred
    assert len(y_true) == len(y_pred)

    cardinality = tf.data.experimental.cardinality(test_set).numpy()
    dataset_batches = int(cardinality)
    expected_batches = math.ceil(len(y_true) / batch)
    assert dataset_batches == expected_batches

def test_evaluation_report_classes(efficientnet, capsys):
    
    # Capture the printed output
    with capsys.disabled():
        evaluation_report(test_set, efficientnet, classes)

    # Check if the class names appear in the printed output
    captured = capsys.readouterr()
    printed_output = captured.out
    for class_name in classes:
        assert (class_name in printed_output, 
                f"Class {class_name} not found in the classification report")