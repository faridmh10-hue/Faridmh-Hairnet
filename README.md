# HairNet: An Automated Deep Learning-Based Hair Style Detection System by Faridmh
HairNet: An Automated Deep Learning-Based Hair Style Detection System

1. Project Description
HairNet is a Deep Learning-based system (specifically, a Convolutional Neural Network or CNN) designed for the automatic detection and classification of human hair styles from digital images. This system is engineered to learn various hair types such as straight, wavy, curly, and coily from a dataset of labeled images.

By leveraging the CNN's ability to recognize visual patterns like texture, wave, and hair contours, HairNet can provide accurate classification results. This project has the potential for application in several fields, including:

Automated hair style recommendations
AI-powered beauty applications
Automatic editing in makeover software

2. Objectives and Benefits
Objectives:
To build an automatic hair style classification system based on CNN.
To train a deep learning model to recognize various hair styles/types from human images.
To provide accurate predictions regarding hair styles from image inputs.
To deliver a system that can be further developed for features like hair style recommendations or AI makeovers.

Benefits:
Enhances user experience in choosing hair styles.
Improves efficiency in hair style search and recommendations.
Supports the development of beauty and fashion applications.
Applies deep learning technology in image recognition.

3. Problem Limitations
Hair Style Types: Only classifies five hair types: Curly, Dreadlocks, Kinky, Straight, and Wavy.
Dataset: Uses images with a resolution of 150 x 150 pixels and good lighting quality.
Classification Method: Utilizes Convolutional Neural Networks (CNNs) for classifying hair images into the five categories mentioned above.
Data Preprocessing: Images will be processed through resizing, augmentation (flip, rotation), and pixel normalization before being used for training.
System Output: The system only classifies hair style images without detecting other attributes like hair thickness or length.
Computation: The system will be optimized to run on hardware with moderate specifications.

4. System Workflow
Start
The system begins by receiving image input from the user or a pre-provided data source.

Hair Image (Input)
The hair image to be processed and classified. This image can come from a prepared dataset.

Preprocessing
Image Resizing: Input images will be resized to match the dimensions accepted by the model (e.g., 150 x 150 pixels).
Image Augmentation: This process involves randomly modifying images, such as horizontal flips, zooms, rotations, and color changes, to increase data variation and reduce overfitting.
Pixel Normalization: Image pixel values will be normalized to be within the range [0, 1] by dividing the pixel values by 255.

CNN Model (Training/Testing)
Convolutional Layers: The model uses several convolutional layers to extract important features from the images.
ReLU Activation: Utilizes the ReLU (Rectified Linear Unit) activation function to introduce non-linearity into the model.
MaxPooling: This operation is used to reduce image dimensions while retaining important features.
Fully Connected Layers: After features are extracted, the images are passed to fully connected layers for classification.
Softmax Output: The softmax activation function will be used in the final layer to generate probabilities for each class (e.g., hair type: straight, wavy, curly, etc.).

5. Output
Hair Style Prediction: Once the model completes its inference, the system will provide a predicted hair type label based on the processed image (e.g., "Wavy," "Straight," "Curly," etc.).
Accuracy: The system will display the model's accuracy to indicate how well it performs predictions based on training and testing data.
Result Visualization: Results can be visualized by displaying the input image with its predicted label (e.g., a hair image with the label "Straight").
It also provides hair care recommendations based on the identified hair type.
End
The process is complete. The system can repeat the process if new images need to be processed.

Dataset
For this project, we utilize a pre-existing dataset sourced from Kaggle:
https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset

Project Files
Our project includes two key Jupyter Notebook (.ipynb) files:

FaridMh_4TIE_HairnetCNN.ipynb: This notebook is dedicated to creating and training the deep learning model. It covers all aspects from data loading and preprocessing to model architecture definition, training, and evaluation.

HairnetCNNDEMO_FaridMuhammadHidayat_4TIE.ipynb: This notebook facilitates the web-based demonstration of our HairNet system. It includes the code for loading the trained model and features a temporary deployment using Gradio, allowing users to interact with the system via a web interface.
