# Script for the classification workflow using the configured settings.

from classificationmodel.dataset import (dataset_generator, 
                                         image_visualization)
from classificationmodel.model import (build_model, 
                                       load_model_weights,
                                       train_model, 
                                       save_model, 
                                       plot_history)
from classificationmodel.evaluation import (evaluate_model, 
                                            evaluation_report)
import configparser
# Specify the absolute file path of the configuration file
config_file_path = 'classificationmodel/config.ini'

# Load the configuration file to get the parameters and paths
config = configparser.ConfigParser()
config.read(config_file_path)
img_size = config.getint('setting', 'img_size')
classes = config.get('setting', 'classes').split(',')
num_classes = config.getint('setting', 'num_classes')
batch = config.getint('setting', 'batch')
epochs = config.getint('setting', 'epochs')
train_params = {'label_mode': config.get('setting', 'label_mode'),
                'color_mode': config.get('setting', 'color_mode'),
                'batch_size': config.getint('setting', 'batch'),
                'image_size': eval(config.get('setting', 'image_size')),
                'seed': config.getint('setting', 'seed')}
img_path = config.get('path', 'img_path')
test_img_path = config.get('path', 'test_img_path')
weight_path = config.get('path', 'weight_path')
model_path = config.get('path', 'model_path')

#Create the datasets to classify and visualize a batch of images.
train_set, val_set, test_set = dataset_generator(img_path,
                                                 test_img_path,
                                                 train_params)
image_visualization(train_set, classes)

#Build the classification model and load into it pre-trained weights.
#Train it, save it for future application and visualize its history.
efficientNet = build_model(num_classes)
loaded_efficientNet = load_model_weights(efficientNet, weight_path)
history, trained_model = train_model(loaded_efficientNet, 
                                     train_set, 
                                     val_set, 
                                     batch, 
                                     epochs, 
                                     weight_path)
saved_model = save_model(efficientNet, model_path)
plot_history(history)

#Use a separate dataset to evaluate the model on its loss, accuracy
#and classification report from sklearn
evaluation_loss, evaluation_accuracy = evaluate_model(trained_model, test_set)
final_report = evaluation_report(test_set, efficientNet, classes)