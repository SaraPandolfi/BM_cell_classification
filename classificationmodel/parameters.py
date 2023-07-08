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
    weight_path (str): Path to the .h5 file with pre-trained weights

"""

img_path = 'images'
test_img_path = 'test_images'
img_size = 300
batch = 8
epochs = 10
classes = ['BLA', 'EBO', 'MMZ', 'NGS']
num_classes = 4

train_params = {
'label_mode': 'categorical',
'color_mode' : 'rgb',
'batch_size' : batch,
'image_size' : (img_size, img_size),
'seed' : 42    
}

weight_path = 'best_efficientnet.h5'