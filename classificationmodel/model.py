import tensorflow as tf
import os
from keras.optimizers import Adam
from keras.applications.efficientnet import EfficientNetB3
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

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

def train_model(model, train_set, val_set, BATCH, EPOCHS):
    """
    Checks if a trained model already exists in the folder and if so,
    it loads the saved weights.
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
        BATCH (int): Batch size.
        EPOCHS (int): Number of epochs.
    
    Returns:
        Tuple containing the training history and the trained model.
    """
    model_path = 'best_efficientnet.h5'    
    if os.path.exists(model_path):
        print('Loading pre-trained model from file...')
        model.load_weights(model_path)

    checkpoint = ModelCheckpoint(model_path,
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
                        batch_size=BATCH, 
                        epochs=EPOCHS, 
                        steps_per_epoch=len(train_set), 
                        validation_data=val_set,
                        validation_steps=len(val_set), 
                        callbacks=[checkpoint, reduce_lr])

    model.callbacks = [checkpoint, reduce_lr]

    return history, model

def save_model(model, filepath):
    """
    Saves the specified model to the specified binary file path using pickle.
    
    Parameters:
        model (Any): Object to save.
        filepath (str): File path to save the object to.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

def load_model(filepath):
    """
    Loads a binary object from the specified file path using pickle.
    
    Parameters:
        filepath (str): File path to load the object from.
    
    Returns:
        Any: Loaded object.
    """
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
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