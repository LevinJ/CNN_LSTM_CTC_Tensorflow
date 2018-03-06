# CNN_LSTM_CTC_Tensorflow

The images are first processed by a CNN to extract features, then these extracted features are fed into a LSTM for character recognition.

CNN+LSTM+CTC based OCR(Optical Character Recognition) implemented using tensorflow. 


I trained a model with 80k images using this code and got 99.98% accuracy on test dataset (20k images). The images in both dataset:

![](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/blob/master/data/ico1-608634b7cb.png)

![](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/blob/master/data/ico2-19c9d50d82.png)

## Overview

This project is based on the great work from [here](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow)

Below improvements are made:
1. correct the time step direction  
Previously the time step direction is channle, which is incorrect. Now it has been corrected to the width direction. see [here](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/issues/8) for more discussion on this issue.
2. optimize trainig scripts  
Previously all training images are loaded into memroy, now a simple image generator is used to generate training batch.
3. metrics implemetation
implement the character and word accuracy in tensorflow.

## Dataset

please see this [issue](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/issues/2) about dataset， the lable file (a .txt file) is in the same folder with images after extracting .tar.gz file.



## Prerequisite

1. TensorFlow 1.4

2. Numpy


  
# Train the model.
python ./train_model.py

# Inference
python ./eval_model.py

