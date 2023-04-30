img_path = 'images'
IMG_SIZE = 250
BATCH = 8
EPOCHS = 5
classes = ['BLA', 'EBO', 'MMZ', 'NGS']
num_classes = 4

augmentation_params = {
    'rotation_range': 0,
    'width_shift_range': 0.,
    'height_shift_range': 0.,
    'shear_range': 0.,
    'zoom_range': 0.,
    'horizontal_flip': False,
    'vertical_flip': False,
    'fill_mode': 'nearest',
    'cval': 0.,
    'validation_split': 0.10
}

train_params = {
    'target_size': (IMG_SIZE, IMG_SIZE),
    'color_mode': 'rgb',
    'class_mode': 'categorical',
    'batch_size': BATCH,
    'shuffle': True,
    'seed': 42
}

