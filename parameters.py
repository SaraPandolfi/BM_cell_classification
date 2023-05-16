img_path = 'images'
test_img_path = 'test_images'
IMG_SIZE = 300
BATCH = 8
EPOCHS = 5
classes = ['BLA', 'EBO', 'MMZ', 'NGS']
num_classes = 4

train_params = {
'label_mode': 'categorical',
'color_mode' : 'rgb',
'batch_size' : BATCH,
'image_size' : (IMG_SIZE, IMG_SIZE),
'seed' : 42    
}

