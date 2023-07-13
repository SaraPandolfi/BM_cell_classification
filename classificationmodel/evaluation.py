import numpy as np
from sklearn.metrics import classification_report
import sys

def evaluate_model(model, test_dataset, output_evaluation):
    """
    Evaluates the model on the test set, gets the predictions
    and returns metrics.

    Parameters:
        model (keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
        output_evaluation (str): Path to .txt file to write into
        the values of loss and accuracy.

    Returns:
        Tuple containing the loss and accuracy.
    """    
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    with open(output_evaluation, 'w') as file:
        sys.stdout = file
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        sys.stdout = sys.__stdout__ 

    return loss, accuracy

def evaluation_report(test_dataset, model, classes, output_report):
    """
    Stores the true and predicted labels to calculate,
    print and saves in a .txt file the classification report of sk.learn.

    Parameters:
        model (keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
        classes (list): List of class names.
        output_report (str): Path to .txt file to write into the report.
    
    Returns: 
        Tuple containing the dict and str of the Classification Report.
    """
    y_true = []
    y_pred = []
    for x, y in test_dataset:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred_probs = model.predict(x)
        y_pred.extend(np.argmax(y_pred_probs, axis=1))
    
    print(classification_report(y_true, y_pred, target_names=classes))

    with open(output_report, 'w') as file:
        sys.stdout = file
        print(classification_report(y_true, y_pred, target_names=classes))
        sys.stdout = sys.__stdout__  

    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict= True)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict= False)
    return report_dict, report