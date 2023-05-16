# Bone Marrow cells classification
This project provides a multiclass classification of four bone marrow cells classes: *Blast* (BLA), *Erythroblast* (EBO), *Metamyelocyte* (MMZ) and *Segmented Neutrophils* (NGS), by using a neural network.
The data used for this project have been downloaded from the open access dataset stored online at the link https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770, used for the research studies pubblished in the plenary paper "Highly accurate differentiation of bone marrow cell morphologies using deep neural networks on a large image data set" of Christian Matek, Sebastian Krappe, Christian Munzenmayer, Torsten Haferlach and Carsten Marr pubblished in 2021, stored at the link http://ashpublications.org/blood/article-pdf/138/20/1917/1845796/bloodbld2020010568.pdf 


Due to the large amount of data stored in this dataset, only 1000 images from each of the four classes mentioned above have been used for the training and validation of the network, while a total of 100 images, equally subdivided between the classes have been used to evaluate the classification performed. The folders, subfolders and IDs are stored in the file *images_IDs*.
## Project organization
The project has been subdivided in the following modules:
1. Database creation
2. Neural network implementation
3. Metrics and results of the model
