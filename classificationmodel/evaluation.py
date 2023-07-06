import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, test_dataset, classes):
    """
    Evaluates the model on the test set, gets the predictions, and returns metrics.

    Parameters:
        model (keras.Model): Trained model.
        test_dataset (tf.data.Dataset): Test dataset.
        classes (list): List of class names.

    Returns:
        Tuple containing the loss and accuracy.
    """    
    loss, accuracy = model.evaluate(test_dataset)
    
    #Store the true and predicted labels to calculate the classification report
    y_true = []
    y_pred = []
    for x, y in test_dataset:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred_probs = model.predict(x)
        y_pred.extend(np.argmax(y_pred_probs, axis=1))
    
    print(classification_report(y_true, y_pred, target_names=classes))
    return loss, accuracy

