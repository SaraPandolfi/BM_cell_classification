import tensorflow as tf
import numpy as np
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


def dataset_generator(img_path, test_img_path, train_params):
    """
    Takes the path to a directory and generates batches of data.
    
    Create a tf.data.Dataset yielding tuples of (x, y) with shape 
    (batch_size, height, width, channels) where x is the input image
    and y is the ground-truth.
    
    The data generation and its split are performed using train_params.
    They give information regarding the image dimensions, color mode,
    batch size, classes, shuffle, and seed.
    
    Parameters:
        img_path (str): Path for the training and validation images directory.
        test_img_path (str): Path for the test images directory.
        train_params (dict): Dictionary containing labels and images color mode,
            size of the batches of data, size of the images, and seed for
            randomness control.
    
    Returns:
        A tf.data.Dataset object yielding tuples of (x, y) where:
        - x is a numpy array containing a batch of images with shape
          (batch_size, height, width, channels).
        - y is a numpy array of corresponding labels.
    """
    train_img_generator = image_dataset_from_directory(img_path,    
                                                       shuffle=True,
                                                       validation_split=0.10,  
                                                       subset= 'training',
                                                       **train_params)
    
    val_img_generator = image_dataset_from_directory(img_path,
                                                     shuffle=True,
                                                     validation_split=0.10,
                                                     subset='validation',
                                                     **train_params)
    
    test_dataset = image_dataset_from_directory(test_img_path,
                                                shuffle=False,
                                                **train_params)

    return train_img_generator, val_img_generator, test_dataset

def image_visualization(dataset, classes):
    """
    Visualizes a sample of images from a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset containing images.

    Returns:
        None

    """
    plt.figure(figsize=(12, 8))
    # select images only from the first batch
    for images, labels in dataset.take(1):
        # convert labels tensor to NumPy array
        # and then one-hot encoded vectors to integer labels
        labels = labels.numpy()
        labels = np.argmax(labels, axis=1)
        for i in range(8):
            ax = plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classes[labels[i]])
            plt.axis("off")