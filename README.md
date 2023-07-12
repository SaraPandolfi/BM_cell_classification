# Bone Marrow cells classification
This project provides a multiclass classification of four bone marrow cells classes: *Blast* (BLA), *Erythroblast* (EBO), *Metamyelocyte* (MMZ) and *Segmented Neutrophils* (NGS), by using a neural network.
The data used for this project have been downloaded from the open access [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770), used for the research studies pubblished in the plenary paper "Highly accurate differentiation of bone marrow cell morphologies using deep neural networks on a large image data set" of Christian Matek, Sebastian Krappe, Christian Munzenmayer, Torsten Haferlach and Carsten Marr pubblished in 2021, stored [here](http://ashpublications.org/blood/article-pdf/138/20/1917/1845796/bloodbld2020010568.pdf).

Table of contents
- Prerequisites
- Installation
- Images download
- Usage
- Testing
- Organization


## Prerequisites
- python>=3.8
- tensorflow>=2.9.1
- numpy>=1.23.1
- scikit-learn>=1.0.1
- matplotlib>=3.4.3

## Installation

To install and run the program clone the repository BM_cell_classification and use pip:

    git clone https://github.com/SaraPandolfi/BM_cell_classification
    cd BM_cell_classification
    pip install -r requirements.txt

The program does not require any installation. You can use it by running the provided script:

    python script.py

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
To use the classification model on the built images folders the following steps can be performed:
1. Parameters setting:
    The parameters stored in the config.ini file can be used directly or modified according to the user's requests.
2. Program execution:
    The classification is performed by executing the [script.py](https://github.com/SaraPandolfi/BM_cell_classification/blob/master/script.py)
3. Model results:
    The resulting statistics of the model will be printed out both in the terminal and in the files indicated in the config.ini file as 'output_evaluation' and 'output_report'. Moreover the saved weights will be found at the path 'best_efficientnet.h5' and the trained model at 'model.json'.
    
## Testing

To test the classification model it can be used pytest testing tool. The tests are in the folder `tests`, and the following lines will run the tests:

    pip install pytest-cov
    cd BM_cell_classification
    pytest --cov=classificationmodel tests/  


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