# Bone Marrow cells classification
This project provides a multiclass classification of four bone marrow cells classes: *Blast* (BLA), *Erythroblast* (EBO), *Metamyelocyte* (MMZ) and *Segmented Neutrophils* (NGS), by using a neural network.
The data used for this project have been downloaded from the open access [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770), used for the research studies pubblished in the plenary paper "Highly accurate differentiation of bone marrow cell morphologies using deep neural networks on a large image data set" of Christian Matek, Sebastian Krappe, Christian Munzenmayer, Torsten Haferlach and Carsten Marr pubblished in 2021, stored [here](http://ashpublications.org/blood/article-pdf/138/20/1917/1845796/bloodbld2020010568.pdf).


Due to the large amount of data stored in this dataset, only 1000 images from each of the four classes mentioned above have been used for the training and validation of the network, while a total of 100 images, equally subdivided between the classes have been used to evaluate the classification performed. The folders, subfolders and IDs are stored in the file *images_IDs*.
## Project overview
The project has been subdivided in the following modules:
1. Database creation
2. Neural network implementation
3. Metrics and results of the model

## Installation

To install the program clone the repository BM_cell_classification and use pip:

    git clone https://github.com/SaraPandolfi/BM_cell_classification
    cd BM_cell_classification
    pip install -r requirements.txt



## Images Download

In order to be able to classify the images they need to be dowloaded first from the [database](/https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) stored online. It is not necessary to download all the dataset, but only the classes used for this purpose. 
1. Access the dataset website.
2. Navigate to the class BLA, EBO, MMZ and NGS and download only their `0001-1000` subfolders.
3. Once the datasets per each class are downloaded, copy them in a `images` folder, in order to have the following structure organized in subfolders BLA, EBO, MMZ, and NGS. The folder structure should look like this:

- `images/`
    - `BLA/0001-1000` - Contains 1000 images belonging to class BLA.
    - `EBO/0001-1000` - Contains 1000 images belonging to class EBO.
    - `MMZ/0001-1000` - Contains 1000 images belonging to class MMZ.
    - `NGS/0001-1000` - Contains 1000 images belonging to class NGS.

4. Repeating the step 2 and 3 per each class in the subfolders `2001-3000`, but this time downloading only the first 100 images per each class, another folder called `test_images` shall be created with the same structure. This smaller folder will be used in the evaluation module.

The images IDs used are reported in the file [images_IDs](https://github.com/SaraPandolfi/BM_cell_classification/blob/master/images_IDs.txt).


