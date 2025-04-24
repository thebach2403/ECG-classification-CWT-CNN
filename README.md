# ECG-classification-CWT-CNN

This project implements an ECG classification pipeline using Continuous Wavelet Transform (CWT) and Convolutional Neural Networks (CNN). The steps include preprocessing the raw ECG data, transforming it into scalogram images using CWT, splitting the data into training and testing sets, defining a CNN model, and training and evaluating the model.

References
Research Paper: "Automatic ECG Classification Using Continuous Wavelet Transform and Convolutional Neural Network" by Tao Wang, Changhua Lu, Yining Sun, Mei Yang, Chun Liu, and Chunsheng Ou.

GitHub Reference: ECG-Classification-Using-CNN-and-CWT

Project Structure:

1. preprocessing.py
Description: This script processes the raw MIT-BIH ECG data. It performs the following tasks:

Baseline wandering removal.

Invalid label filtering.

R-peak alignment and signal normalization.

Conversion of ECG beats into scalogram images using Continuous Wavelet Transform (CWT).

Output: A folder containing all the scalogram images for each ECG beat.

2. train_test_split.py
Description: This script accesses the folder containing the scalogram images generated in the preprocessing.py step and splits the data into train and test sets. It creates two new folders: train_images and test_images containing the corresponding images for each set.

3. Class_Model.py
Description: This file defines the class for the Convolutional Neural Network (CNN) model that will be used for ECG classification. The model is designed to accept scalogram images and classify them into different categories based on the type of heartbeats.

4. Class_Dataset.py
Description: This script contains the class definition for the dataset used in the training process. It handles loading images from the train_images and test_images folders, as well as preprocessing them to be ready for CNN input.

5. training.py
Description: This script handles the training process. It loads the dataset, initializes the CNN model, and trains the model using the training data.

6. test.py
Description: This script is used to evaluate the trained model. It uses the test dataset to predict classifications and then evaluates the performance using accuracy and other metrics.

