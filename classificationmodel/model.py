import tensorflow as tf
import os
from keras.optimizers import Adam
from keras.applications.efficientnet import EfficientNetB3
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

def build_model(num_classes):
    """
    Creates an EfficientNetB3 keras model with the specified number of
    classes to classify.
    
    Parameters:
        num_classes (int): Number of classes to classify.
    
    Returns:
        EfficientNetB3 keras model.
    """    
    model = EfficientNetB3(include_top=True,
                           weights=None,
                           input_tensor=None,
                           input_shape=None,
                           pooling=None,
                           classes= num_classes,
                           classifier_activation='softmax')
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss= categorical_crossentropy,
                  metrics=['accuracy'])
    
    return model

def load_model_weights(model, weight_path):
    """
    Loads pre-trained model weights from a file.
    If the weights file exists, it loads the weights into the given model and returns the model.
    Otherwise, it prints a message indicating that no pre-trained weights were found and returns None.

    Parameters:
        model (keras.Model): Model to load the weights into.
        weight_path (str): Path to the pre-traned saved weights.

    Returns:
        keras.Model or None: Loaded model if the weights were loaded successfully, None otherwise.
    """
    if os.path.exists(weight_path):
        print('Loading pre-trained model from file...')
        model.load_weights(weight_path)
        return model
    else:
        print('No pre-trained model weights found.')
        return False

def train_model(model, train_set, val_set, batch, epochs, weight_path):
    """
    Defines the keras callbacks by ModelCheckpoint monitoring the accuracy,
    and by ReduceLROnPlateau monitoring the loss.
    Trains the model on the training set storing the keras fit
    in the variable history.
    Evaluates it on the validation set.
    Adds the callbacks attribute to the model object.
    The trained model is saved to the file specified by `model_path`.
    
    Parameters:
        model (keras.Model): Model to train.
        train_set (tf.data.Dataset): Training dataset.
        val_set (tf.data.Dataset): Validation dataset.
        batch (int): Batch size.
        epochs (int): Number of epochs.
        weight_path (str): Path to the .h5 file to store the updated weights.
    
    Returns:
        Tuple containing the training history and the trained model.
    """     
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_accuracy',
                                 save_best_only=True, 
                                 mode='max', 
                                 verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.2, 
                                  patience=2, 
                                  min_lr=1e-7, 
                                  verbose=1)

    history = model.fit(train_set, 
                        batch_size=batch, 
                        epochs=epochs, 
                        steps_per_epoch=len(train_set), 
                        validation_data=val_set,
                        validation_steps=len(val_set), 
                        callbacks=[checkpoint, reduce_lr])

    model.callbacks = [checkpoint, reduce_lr]

    return history, model

def save_model(model, model_path):
    """
    Saves the specified model to the specified JSON file path.
    
    Parameters:
        model (keras.Model): Model to save.
        model_path (str): File path to save the model to.
    """
    model_json = model.to_json()
    with open(model_path, 'w') as file:
        file.write(model_json)

def load_model(model_path):
    """
    Loads a model from the specified JSON file path and compiles it.
    
    Parameters:
        model_path (str): File path to load the model from.
    
    Returns:
        keras.Model: Loaded model.
    """
    with open(model_path, 'r') as file:
        model_json = file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss= categorical_crossentropy,
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """
    Plots and saves the training and validation accuracy and loss over epochs.
    
    Parameters:
        history (keras.callbacks.History): History object returned by
        the `fit` method of a Keras model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation accuracy values
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')

    # Plot training & validation loss values
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    plt.savefig('accuracy_loss.png')
    plt.show()