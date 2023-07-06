import os
import sys
import pytest
import numpy as np

# Get the current file's absolute path and move backward to 
# add the directory to the Python module search path
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
os.chdir(grandparent_dir)
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from classificationmodel.model import load_model, build_model
from classificationmodel.dataset import dataset_generator
from classificationmodel.parameters import (img_path,
                                            test_img_path, 
                                            train_params, 
                                            num_classes,
                                            classes)
from classificationmodel.evaluation import evaluate_model

_, _, test_set = dataset_generator(img_path,
                                   test_img_path,
                                   train_params)

def test_evaluate_model():
    """
    This test loads or builds the model and the function evaluate_model
    and verifies that the loss and accuracy are within a certain tolerance
    close to the actual values.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        model = build_model(num_classes)
    except Exception as e:
        pytest.fail(f"Failed to load or build model: {e}")
    
    test_loss, test_accuracy = evaluate_model(model, 
                                              test_set, 
                                              classes)
    expected_loss = 1.10  
    expected_accuracy = 0.8  
    tolerance = 1e-1 
    assert (np.abs(test_loss - expected_loss) < tolerance,
            f"Test Loss does not match expected value. "
            f"Expected: {expected_loss}, Actual: {test_loss}")
    assert (np.abs(test_accuracy - expected_accuracy) < tolerance,
            f"Test Accuracy does not match expected value. "
            f"Expected: {expected_accuracy}, Actual: {test_accuracy}")