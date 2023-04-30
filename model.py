import tensorflow as tf
import os
from keras.optimizers import Adam
from keras.applications.efficientnet import EfficientNetB3
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from parameters import num_classes, BATCH, EPOCHS
from dataset import train_set, val_set
import pickle


def build_model(num_classes):
    '''
    Creates an EfficientNetB3 keras model with the specified number of classes to classify.
    Parameter
    ---------
    num_classes: int
        number of classes to classify.
    
    Returns
    ---------
    EfficientNetB3 keras model

    '''
    
    model = EfficientNetB3(
        include_top=True,
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
    '''
    Checks if a trained model already exists in the folder and if so it loads the saved weight;
    defines the keras callbacks through ModelCheckpoint monitoring the accuracy, and
    through ReduceLROnPlateau monitoring the loss; 
    trains the model on the training set storing the keras fit in the variable history;
    evaluates it on the validation set;
    adds the callbacks attribute to the model object.
    
    Parameters
    ----------
    model: keras.Model
        model to train
    train_set: tf.data.Dataset
        training dataset
    val_set: tf.data.Dataset
        validation dataset
    BATCH: int
        batch size
    EPOCHS: int
        number of epochs
    
    Returns
    --------
    Tuple containing the training history and the trained model
    
    '''


    model_path = 'best_efficientnet.h5'
    
    if os.path.exists(model_path):
        print('Loading pre-trained model from file...')
        model.load_weights(model_path)


    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

    history = model.fit(train_set, epochs=EPOCHS, steps_per_epoch=len(train_set), validation_data=val_set,
                        validation_steps=len(val_set), callbacks=[checkpoint, reduce_lr])

    model.callbacks = [checkpoint, reduce_lr]

    return history, model


def save_model(model, filepath):
    '''
    Saves the specified model to the specified file path using pickle.
    Parameters
    ---------
    model: Any
        object to save
    filepath: str
        file path to save the object to
    '''
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def load_model(filepath):
    '''
    Loads an object from the specified file path using pickle.
    Parameter
    ---------
    filepath: str
        file path to load the object from
    
    Returns
    -------
        Any: loaded object
    '''
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


efficientNet = build_model(num_classes)
model_history, trained_model = train_model(efficientNet, train_set, val_set, BATCH, EPOCHS)
saved_model = save_model(efficientNet, 'model.pkl')







