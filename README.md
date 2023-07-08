# Bone Marrow cells classification
This project provides a multiclass classification of four bone marrow cells classes: *Blast* (BLA), *Erythroblast* (EBO), *Metamyelocyte* (MMZ) and *Segmented Neutrophils* (NGS), by using a neural network.
The data used for this project have been downloaded from the open access [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770), used for the research studies pubblished in the plenary paper "Highly accurate differentiation of bone marrow cell morphologies using deep neural networks on a large image data set" of Christian Matek, Sebastian Krappe, Christian Munzenmayer, Torsten Haferlach and Carsten Marr pubblished in 2021, stored [here](http://ashpublications.org/blood/article-pdf/138/20/1917/1845796/bloodbld2020010568.pdf).

Table of contents
- Prerequisites
- Installation
- Images download
- Usage
- Organization


## Prerequisites
- python>=3.8
- tensorflow>=2.9.1
- numpy>=1.23.1
- scikit-learn>=1.0.1
- matplotlib>=3.4.3

To install the program clone the repository BM_cell_classification and use pip:

    git clone https://github.com/SaraPandolfi/BM_cell_classification
    cd BM_cell_classification
    pip install -r requirements.txt

## Installation
The program does not require any installation. You can use it by running your scripts and notebooks inside the BM_cell_classification folder directly.

## Images Download

In order to be able to classify the images they need to be dowloaded first from the [database](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) stored online. It is not necessary to download all the dataset, but only the classes used for this purpose. Here there are the instruction to create both the `images` and the `test_images` folders. The last one is used for validating the model.

1. Access the dataset website, scroll down until the download section, press the download button.
2. Download the required extensions "IBM Aspera Connect".
3. Click into the folder "BM_cytomorphology_data".
4. Navigate to the class BLA, EBO, MMZ and NGS and download only their `0001-1000` subfolders respectively.
3. Once the datasets per each class are downloaded, copy them in a `images` folder, in order to have the following structure organized in subfolders BLA, EBO, MMZ, and NGS. The folder structure should look like this:

- `images/`
    - `BLA/0001-1000` - Contains 1000 images belonging to class BLA.
    - `EBO/0001-1000` - Contains 1000 images belonging to class EBO.
    - `MMZ/0001-1000` - Contains 1000 images belonging to class MMZ.
    - `NGS/0001-1000` - Contains 1000 images belonging to class NGS.

4. Repeating the step 2 and 3 per each class in the subfolders `2001-3000`, but this time download only the first 100 images per each class, another folder called `test_images` shall be created with the same structure. This smaller folder will be used in the evaluation module as a test set.

The images IDs used are reported in the file [images_IDs](https://github.com/SaraPandolfi/BM_cell_classification/blob/master/images_IDs.txt).


## Usage

1. Dataset preparation:
```python
    #import the function and from config.ini the paths to the images folders and for the dataset setup
    from classificationmodel.dataset import dataset_generator
    #Get the parameters
    config = configparser.ConfigParser()
    config.read('config.ini')
    img_path = config.get('path', 'img_path')
    test_img_path = config.get('path', 'test_img_path')
    train_params = {
    'label_mode': config.get('setting', 'label_mode'),
    'color_mode': config.get('setting', 'color_mode'),
    'batch_size': config.getint('setting', 'batch'),
    'image_size': eval(config.get('setting', 'image_size')),
    'seed': config.getint('setting', 'seed')}

    #build the datasets for training, validation and final testing
    train_set, val_set, test_set = dataset_generator(img_path,
                                                 test_img_path,
                                                 train_params)
```
2. Model training
```python
    #import the functions to create, train, save the model and finally plot its history
    #import the parametes for the training of the model
    from classificationmodel.model import build_model, train_model, save_model, plot_history
    batch = config.getint('setting', 'batch')
    epochs = config.getint('setting', 'epochs')
    num_classes = config.getint('setting', 'num_classes')
    weight_path = config.get('path', 'weight_path')

    #build the efficientNet model
    efficientNet = build_model(num_classes)

    #train the model, store the history and the trained model
    model_history, trained_model = train_model(efficientNet,
                                                train_set, 
                                                val_set, 
                                                batch, 
                                                epochs,
                                                weight_path)
    
    #save the model for future applications
    saved_model = save_model(efficientNet, 'model.json')

    #plot the histry as accuracy and loss over the epochs
    plot_history(model_history)
```
3. Model evaluation
```python
    #import the function to evaluate the model and print the classification report of sklearn
    #import the name of the classes classified from the parameters
    from classificationmodel.evaluation import evaluate_model
    classes = config.get('setting', 'classes').split(',')

    #evaluate the model and print the classification report, loss and accuracy
    test_loss, test_accuracy = evaluate_model(trained_model, test_set)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')    
    evaluation_report(test_set, efficientNet, classes)
```
## Organization

The project has been subdivided as follows:
- `classificationmodel/`
    - dataset.py - module to create the train and test dataset from specific folders
    - model.py - module to create and train the model
    - evaluate.py - module to evaluate the trained model by the classification report of sklearn 
    - config.ini - configuration file to use by configparser
- `tests/` - test routine, one per each module, with own parameters file

Due to the large amount of data stored in this dataset, only 1000 images from each of the four classes mentioned above have been used for the training and validation of the network, while a total of 100 images, equally subdivided between the classes have been used to evaluate the classification performed. The folders, subfolders and IDs are stored in the file *images_IDs*.
I suggest to try the classification task on Google Colab since the execution time could be long in an average level laptop.