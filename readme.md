## Pneumonia Detection Using Chest X-Ray Images
This project detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model is trained on the Chest X-ray Images (Pneumonia) dataset to classify images as either NORMAL or PNEUMONIA.


## Dataset
The dataset used for this project is the Chest X-ray Images (Pneumonia) dataset, available on Kaggle. It contains images categorized into two classes: NORMAL and PNEUMONIA.

## Requirements
To run this project, you need Python 3.x and the dependencies listed in requirements.txt. Install the required packages with:

                  pip install -r requirements.txt


## Training the Model
This project uses a custom Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The model is trained using the Chest X-ray Images (Pneumonia) dataset.

Run the train.py script to start training the model.

                  python scripts/train.py

The model will be saved to the model directory as saved_model.h5.


## Testing the Model
After training the model, you can evaluate its performance on the test dataset using the test.py script.

Run the test.py script to evaluate the model’s accuracy on the test dataset. 

                    python scripts/test.py

This will print the test accuracy and example predictions from the test dataset.


## Model Architecture
The model consists of multiple Convolutional Neural Network (CNN) layers, followed by MaxPooling, Flattening, and Dense layers. The architecture also includes Dropout layers to prevent overfitting. The model uses the binary cross-entropy loss function and the Adam optimizer.


## Architecture Summary:
1. Conv2D layers for feature extraction
2. MaxPooling2D layers for downsampling
3. Flatten layer to convert 2D features to 1D
4. Dense layers for final classification
5. Dropout layer to reduce overfitting