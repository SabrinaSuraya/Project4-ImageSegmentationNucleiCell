# Project4-ImageSegmentationNucleiCell
To get the mask of the image segmentation of cell nuclei
link: https://www.kaggle.com/c/data-science-bowl-2018

# Summary
To make image segmentation of cell nuclei
deal with a large dataset
the deep learning model is used and trained
The model use Transfer Learning from mobilenetV2
IDE and Framework
The project built with Spyder as the main IDE
use Tensorflow, Keras, Numpy, Mathplot
# Methodology
The folder contain 2; train folder ( 603 images) and test (63 images) folder; 
each folder contain the image folder and mask folder for all teh image of nuclei cell

Here some example of image and mask in the first 3 image in train
![image](https://user-images.githubusercontent.com/73817610/175865928-7c37ae06-595c-488c-93c0-0131ad7525be.png)

We do teh data normalization in the mask as the value is varies between 0 to 255.
![image](https://user-images.githubusercontent.com/73817610/175866082-8fe7807f-9e5e-425a-9467-2b7c146f40d8.png)
We want t o change this value into ranging 0 to 1

we did the train test split with 20 of the data is doing the test .

We use the transfer Learning from MobileNetV2 as the pretrained and as feature extractor 
Model summary: image

-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= BinaryCrossentropy', metrics of accuracy, batch_size of 32 and epochs of 200

The value is display by using TensorBoard:
image
image

Predicting a new image image

image


![image](https://user-images.githubusercontent.com/73817610/175869383-3944eff1-2911-4058-a3d4-dcfc8f3b5fc8.png)
![image](https://user-images.githubusercontent.com/73817610/175869678-55bc7b82-bcdd-4ead-9bf7-5bd1be21e925.png)
![image](https://user-images.githubusercontent.com/73817610/175869689-d6cdfefa-583a-49cd-82be-79cd30224087.png)
![image](https://user-images.githubusercontent.com/73817610/175869705-11316a39-4eef-49d4-aa7b-73a1b387eda1.png)
![image](https://user-images.githubusercontent.com/73817610/175869724-e4a8fd16-27f4-4352-892c-187fe46dc761.png)
![image](https://user-images.githubusercontent.com/73817610/175869772-cd58cbb3-5695-4290-982f-d78e56980949.png)
![image](https://user-images.githubusercontent.com/73817610/175869815-7b34afea-a7ae-4a28-9daa-03c50ed58165.png)
![image](https://user-images.githubusercontent.com/73817610/175869845-2d3b339b-417b-40fe-a459-39bd41b4cb44.png)
![image](https://user-images.githubusercontent.com/73817610/175869854-9e56e80e-26ee-4a36-9f9d-770e08f14ceb.png)








