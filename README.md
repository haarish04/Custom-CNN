# Custom-CNN

Custom built CNN architecture with 90%+ accuracy 

The dataset must be divided into train and test

In the train folder, the folders must be named as 0,1,2.. and so on, based on the number of classes
Map each class number to a class name in the "classes" variable and store all images of one class in one numbered folder

eg. animal recognition
classes= {0:'cat',
          1:'dog',
          2.'horse'}

Epochs can be increased depending on the volume of data

The performance.py contains implementation to plot the test and validation accuracy and losses

The user input.py has implementation to give one image as input to the model which can be done after fitting the model
