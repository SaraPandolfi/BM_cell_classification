"""
Module for storing parameters used in the modules.

Parameters:
    img_path (str): Path to the folder containing the training images.
    test_img_path (str): Path to the folder containing the test images.
    IMG_SIZE (int): Size of the images (width and height).
    BATCH (int): Batch size for training.
    EPOCHS (int): Number of training epochs.
    classes (list): List of class labels.
    num_classes (int): Number of classes.
    train_params (dict): Dictionary of parameters for training the model.

"""

img_path = 'images'
test_img_path = 'test_images'
IMG_SIZE = 300
BATCH = 8
EPOCHS = 10
classes = ['BLA', 'EBO', 'MMZ', 'NGS']
num_classes = 4

train_params = {
'label_mode': 'categorical',
'color_mode' : 'rgb',
'batch_size' : BATCH,
'image_size' : (IMG_SIZE, IMG_SIZE),
'seed' : 42    
}
