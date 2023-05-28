import pytest
import numpy as np
from model import load_model
from dataset import test_set
from parameters import classes
from evaluation import evaluate_model

def test_evaluate_model():
    '''
    This test load the module and the function evaluate_module
    and verifies that the  loss and accuracy are within a certain tolerance close to the actual values.
    '''

    model = load_model('model.pkl')

    test_loss, test_accuracy = evaluate_model(model, test_set, classes)

    expected_loss = 1.10  
    expected_accuracy = 0.8  
    tolerance = 1e-1 
    assert np.abs(test_loss - expected_loss) < tolerance, f"Test Loss does not match expected value. Expected: {expected_loss}, Actual: {test_loss}"
    assert np.abs(test_accuracy - expected_accuracy) < tolerance, f"Test Accuracy does not match expected value. Expected: {expected_accuracy}, Actual: {test_accuracy}"


