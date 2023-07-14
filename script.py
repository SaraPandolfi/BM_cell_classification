# Script for the classification workflow using the configured settings.

from classificationmodel.dataset import dataset_generator 
                                         
from classificationmodel.model import (build_model,
                                       train_model, 
                                       save_model, 
                                       plot_history)
from classificationmodel.evaluation import (evaluate_model, 
                                            evaluation_report)
import configparser
import json
# Specify the absolute file path of the configuration file
config_file_path = 'classificationmodel/config.json'

# Load the configuration file
config = configparser.ConfigParser()
# Read the JSON configuration file
with open(config_file_path) as config_file:
    config = json.load(config_file)

# Access configuration values
train_params = config['setting']['train_params']
img_path = config['path']['img_path']
test_img_path = config['path']['test_img_path']
classes = config['setting']['classes']
num_classes = config['setting']['num_classes']
batch = config['setting']['batch']
epochs = config['setting']['epochs']
weight_path = config['path']['weight_path']
model_path = config['path']['model_path']
output_report = config['path']['output_report']
output_evaluation = config['path']['output_evaluation']

#Create the datasets to classify 
train_set, val_set, test_set = dataset_generator(img_path,
                                                 test_img_path,
                                                 train_params)

#Build the classification model and load into it pre-trained weights.
#Train it, save it for future application and visualize its history.
efficientNet = build_model(num_classes)
history, trained_model = train_model(efficientNet, 
                                     train_set, 
                                     val_set, 
                                     batch, 
                                     epochs, 
                                     weight_path)
saved_model = save_model(trained_model, model_path)
plot_history(history)

#Use a separate dataset to evaluate the model on its loss, accuracy
#and classification report from sklearn
evaluation_loss, evaluation_accuracy = evaluate_model(trained_model, test_set, output_evaluation)
evaluation_report(test_set, trained_model, classes, output_report)