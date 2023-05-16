from model import load_model
from dataset import test_set
from parameters import classes
import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, test_dataset, classes):
    '''
    Evaluates the model on the test set, gets the predictions and returns metrics.

    Parameters
    ----------
    model: keras.Model
        trained model
    test_set: tf.data.Dataset
        test dataset
    classes: list
        list of class names

    Returns
    -------
    Tuple containing the loss, accuracy and classification report.
    '''
    
    loss, accuracy = model.evaluate(test_dataset)

    y_true = []
    y_pred = []
    for x, y in test_dataset:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred_probs = model.predict(x)
        y_pred.extend(np.argmax(y_pred_probs, axis=1))

    
    print(classification_report(y_true, y_pred, target_names=classes))

    return loss, accuracy

if __name__ == '__main__':
    model = load_model('model.pkl')

    test_loss, test_accuracy = evaluate_model(model, test_set, classes)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

