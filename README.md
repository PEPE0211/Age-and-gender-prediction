# Real-Time Gender and Age Prediction Using UTKFace Dataset

This project leverages the UTKFace dataset to predict gender and age in real-time using your laptop's camera. It employs a Convolutional Neural Network (CNN) with a multi-output model architecture to simultaneously predict age (as a regression problem) and gender (as a classification problem).

## Model Overview

The model is built using TensorFlow and Keras. It consists of shared convolutional and pooling layers, followed by two branches: one for age prediction and another for gender prediction. The age prediction branch outputs a single continuous value, whereas the gender prediction branch outputs two values corresponding to the predicted gender probabilities.

### Model Architecture

- **Shared Layers:**
  - Conv2D: 32 filters, 3x3, activation='relu'
  - MaxPooling2D: 2x2
  - Conv2D: 64 filters, 3x3, activation='relu'
  - MaxPooling2D: 2x2
  - Flatten
- **Age Prediction Branch:**
  - Dense: 128 units, activation='relu'
  - Dropout: 0.5
  - Dense (age_output): 1 unit
- **Gender Prediction Branch:**
  - Dense: 128 units, activation='relu'
  - Dropout: 0.5
  - Dense (gender_output): 2 units, activation='softmax'

### Training Results

- Final Epoch (20/20):
  - Loss: 84.1137
  - Age Output Loss: 83.9496
  - Gender Output Loss: 0.1641
  - Age Output MAE: 6.8025
  - Gender Output Accuracy: 92.36%
  - Validation Loss: 107.6182
  - Validation Age Output Loss: 107.2527
  - Validation Gender Output Loss: 0.3655
  - Validation Age Output MAE: 7.4744
  - Validation Gender Output Accuracy: 88.80%

## Dataset
This project uses the UTKFace dataset for training the model. The UTKFace dataset is a large-scale face dataset with age, gender, and ethnicity annotations. For more details and to download the dataset visit  [this Kaggle link]([https://susanqq.github.io/UTKFace/](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped) - Download the dataset and explore its documentation.

![Real-Time Prediction Snapshot](https://github.com/PEPE0211/Age-and-gender-prediction/blob/main/snapshot.png)

