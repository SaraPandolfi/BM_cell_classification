import os
import sys
import pytest
import numpy as np
import tensorflow as tf
import math
import configparser
import json


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
evaluation_path = os.path.join(grandparent_dir, config.get('path', 'output_evaluation'))
report_path = os.path.join(grandparent_dir, config.get('path', 'output_report'))

os.chdir(grandparent_dir)
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from classificationmodel.model import (load_model_weights, 
                                       build_model, 
                                       train_model)
from classificationmodel.dataset import dataset_generator
from classificationmodel.evaluation import evaluate_model, evaluation_report

train_set, val_set, test_set = dataset_generator(img_path,
                                                 test_img_path,
                                                 train_params)

@pytest.fixture(scope="module")
def efficientnet():
    """
    This fixture function loads a pre-trained EfficientNet model
    from the using a saved weights file or builds and trains it
    if the file is not found.
    GIVEN:
        - The 'best_efficientnet.h5' file is present or not.
    WHEN:
        - The EfficientNet model is loaded with the weights 
          if the file is present.
    THEN:
        - If the file is not found, the model is built and trained.
        - The model is returned.
    """
    model = build_model(num_classes)
    try:
        model = load_model_weights(model, weight_path)
    except FileNotFoundError:
        _, model = train_model(model,
                               train_set, 
                               val_set, 
                               batch, 
                               epochs, 
                               weight_path)
    except Exception as e:
        pytest.fail(f"Failed to load or build model: {e}")
    return model

def test_evaluate_model(efficientnet):
    """
    This test verifies that the loss and accuracy are within a 
    certain tolerance close to the actual values.
    GIVEN:
        - A trained EfficientNet model.
        - A test dataset.
    WHEN:
        - The evaluate_model function is called with the model and test_set.
    THEN:
        - The calculated loss and accuracy are compared with expected values
          within a tolerance.    
    """      
    test_loss, test_accuracy = evaluate_model(efficientnet, 
                                              test_set,
                                              evaluation_path)
    expected_loss = 0.7
    expected_accuracy = 0.8
    tolerance = 2e-1 
    assert (np.abs(test_loss - expected_loss) < tolerance), (
            f"Test Loss does not match expected value. "
            f"Expected: {expected_loss}, Actual: {test_loss}")
    assert (np.abs(test_accuracy - expected_accuracy) < tolerance), (
            f"Test Accuracy does not match expected value. "
            f"Expected: {expected_accuracy}, Actual: {test_accuracy}")

def test_evaluation_report_lists(efficientnet):
    """
    This test checks the lists of true and predicted labels.
    GIVEN:
        - A trained EfficientNet model.
        - The test_set dataset.
    WHEN:
        - Iterating through the test_set and predicting labels using the model.
    THEN:
        - The shape of the true and predicted labels are check to be equal.
        - The contents of true and predicted labels are checked.
        - The lengths of true and predicted labels are compared.
        - The number of batches in the test sets matches the ones of
          the true labels.          
    """
    y_true = []
    y_pred = []
        
    for x, y in test_set:       
        y_pred_probs = efficientnet.predict(x)
        assert y.shape == y_pred_probs.shape
        assert y.shape[1] == num_classes, "Incorrect shape of 'y' array"
        assert y_pred_probs.shape[1] == num_classes, (
            "Incorrect shape of predicted probabilities") 
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(y_pred_probs, axis=1))

   # Compare the lengths of y_true and y_pred
    assert len(y_true) == len(y_pred)
    dataset_batches = len(test_set)
    expected_batches = math.ceil(len(y_true) / batch)
    assert dataset_batches == expected_batches

def test_evaluation_report_labels(efficientnet):
    """
    This test checks if the generated classification report contains the expected labels.
    GIVEN:
        - A test dataset.
        - A trained model.
        - A list of class names.
        - A path to the .txt file.
    WHEN:
        - The evaluation_report function is called with the given inputs.
    THEN:
        - The generated classification report contains the expected labels.
    """
    expected_labels = classes
   
    report_dict, _ = evaluation_report(test_set, 
                                       efficientnet, 
                                       classes, 
                                       report_path)
    report_labels = report_dict.keys()
    assert set(expected_labels).issubset(report_labels), (
        "Labels are missing in the classification report")
    
def test_evaluation_output(efficientnet):
    """
    This test checks if the values of Loss and Accuracy reported
    in the .txt file are the same as calculated.
    GIVEN:
        - A test dataset.
        - A trained model.
        - A path to the .txt file.
    WHEN:
        - The evaluation_model function is called with the given inputs.
    THEN:
        - The outputs in the file are the same as the calculated ones.
    """
    # Verify the content of the output file
    with open(evaluation_path, 'r') as file:
        content = file.read()
        expected_loss = evaluate_model(efficientnet,
                                       test_set,
                                       evaluation_path)[0]
        expected_accuracy = evaluate_model(efficientnet,
                                           test_set,
                                           evaluation_path)[1]

        assert f'Test Loss: {expected_loss}' in content, (
            "Loss value not found in evaluation output")
        assert f'Test Accuracy: {expected_accuracy}' in content, (
            "Accuracy value not found in evaluation output")
        
def test_evaluation_report(efficientnet):
    """
    This test checks if the values of the classification report in the
    .txt file are the same as calculated ones.
    GIVEN:
        - A test dataset.
        - A trained model.
        - A list of class names.
        - A path to the .txt file.
    WHEN:
        - The evaluation_report function is called with the given inputs.
        - They are converted into str and stripped to avoid differences
          in new lines.
    THEN:
        - The outputs in the file are the same as the calculated ones.
    """
    # Verify the content of the output file
    with open(report_path, 'r') as file:
        content = file.read()
    # Remove leading whitespace and trailing newlines
        content_stripped = str(content).strip()
        _, expected_report = evaluation_report(test_set, 
                                            efficientnet, 
                                            classes, 
                                            report_path)
        expected_report_stripped = str(expected_report).strip()
        assert content_stripped == expected_report_stripped, (
            "Classification report does not match expected output")