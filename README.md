# Project4-ImageSegmentationNucleiCell
To get the mask of the image segmentation of cell nuclei
link: https://www.kaggle.com/c/data-science-bowl-2018

# Summary
- To make image segmentation of cell nuclei
- deal with a large dataset
- the deep learning model is used and trained
- The model use Transfer Learning from mobilenetV2
# IDE and Framework
- The project built with Spyder as the main IDE
- use Tensorflow, Keras, Numpy, Mathplot
# Methodology
- The folder contain 2; train folder ( 603 images) and test (63 images) folder; 
- each folder contain the image folder and mask folder for all teh image of nuclei cell

- Here some example of image and mask in the first 3 image and its mask in train folder

![image](https://user-images.githubusercontent.com/73817610/175865928-7c37ae06-595c-488c-93c0-0131ad7525be.png)

- We do the data normalization in the mask and the image in training data; the value is varies between 0 to 255.

![image](https://user-images.githubusercontent.com/73817610/175866082-8fe7807f-9e5e-425a-9467-2b7c146f40d8.png)

- We want to change these value into ranging 0 to 1

# Model
- we did the train test split with 20% of the data is doing the test.
- we convert the numpy array of x_train, x_test, y_train, y_test into tensor slices.
- we Zip the tensor clices into ZipDataSet of train and test.
- As we deal a large amount of images and masks. we need to conver the train and test data into PrefetchDataSet. This is use to avoid bottle neck when import the images into dataset.
- we create the model; we use pretrained model from MobileNetV2 as base model and as feature extractor.
- make a few modification on the layer as we deal the image segmentation problem. Has a few activations layer as base model output. Has a downstack to define extraction model and freeze the trainable layer. Define the upsampling by using model pix2pix
- the model use modified Unet; use Functional API, Downsampling through the model then upsampling the model and establish teh skip connections between the downsampling and upsampling. The last layer we use Conv2DTranpose:

![image](https://user-images.githubusercontent.com/73817610/176503181-608af9e2-d658-4dbb-8e14-0cb08738e5ec.png)

- We has 3 output class which is the pixel of nuclei cell, pixel bordering the nuclei cell and background pixel. ( in Image Segmentation, each pixel in the image need to assign to the its classes)

-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= SparseCrossentropy', metrics of accuracy, epochs of 20

-This is show how the image, mask  and predicted mask before the model training:

![image](https://user-images.githubusercontent.com/73817610/176503840-0bcd7fe3-6a72-41b9-ac7d-950944a8b185.png)

EPOCHS=1

EPOCHS=20

The value is display by using TensorBoard:


# Model Prediction

Predicting the first 5 images in test 










