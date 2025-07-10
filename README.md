# HairNet: An Automated Deep Learning-Based Hair Style Detection System by Faridmh
HairNet: An Automated Deep Learning-Based Hair Style Detection System

**What is HairNet?**

HairNet is a cool system I built using Deep Learning, specifically a Convolutional Neural Network (CNN). Its main goal is simple: to automatically detect and classify human hair styles from digital images. So, HairNet is trained to be super smart at recognizing various hai r types like straight, wavy, curly, kinky, and even dreadlocks, just from a collection of labeled photos.

How does it do that? HairNet uses the CNN's amazing ability to "see" really detailed visual patterns, such as hair texture, wave patterns, or overall contours. This way, the classification results are really accurate.

And get this, this project has tons of real-world potential, for example, in:

- **Automated Hair Style Recommendations:** Imagine uploading a photo and instantly getting suggestions for suitable hair styles!

- **AI-Powered Beauty Applications:** Great for virtual try-ons or automatic makeup apps.

- **Automatic Editing in Makeover Software:** Makes it much easier to edit hair sections in photo editing applications.

**Objectives and Benefits**

**HairNet's Goals:**

- To build an automatic hair style classification system using CNN technology.

- To train a robust deep learning model that can recognize various hair styles from human images.

- To provide accurate predictions of hair styles from image inputs.

- To lay the foundation for a system that can be further developed for more advanced features like AI-driven makeovers or style recommendations.

**Benefits for Us:**

- **Enhanced User Experience:** Makes choosing hair styles easier and more fun for users.

- **Improved Efficiency:** Saves time and effort in searching for or recommending hair styles.

- **Drives Innovation:** Really helps in developing cutting-edge beauty and fashion applications.

- **Practical AI Application:** Serves as a direct example of how deep learning technology can be used to solve real-world problems, especially in image recognition.

**Recognized Hair Types & Limitations**

**Hair Style Types HairNet Can Classify:**

Currently, HairNet can classify five distinct hair types:

**1. Curly**

**2. Dreadlocks**

**3. Kinky**

**4. Straight**

**5. Wavy**

**System Limitations:**

**- Image Resolution:** This system works best with images that have a resolution of 150 x 150 pixels and good lighting quality.

**- Classification Method:** It strictly uses Convolutional Neural Networks (CNNs) for classification.

**- Output Scope:** It only classifies hair styles; it doesn't detect other attributes like hair thickness or length.

**- Computation:** The system is designed to run smoothly on hardware with moderate specifications.

**System Workflow**

HairNet follows a clear, step-by-step process from when you input an image until it predicts the hair style:

**Explaining the Steps:**

**1. Start:** The system is ready to receive an image.

2. Hair Image (Input): This is the image you want to process and classify, either from you directly or from a prepared dataset.

3. Preprocessing:

- Image Resizing: All images will be resized to 150 x 150 pixels to match the model's required input.

- Image Augmentation: To make the training data more diverse and prevent the model from overfitting (memorizing too much), images are slightly altered, like being horizontally flipped, zoomed, rotated, or having color changes.

- Pixel Normalization: Image pixel values will be scaled to a range of [0, 1] by dividing them by 255.

4. CNN Model (Training/Testing):

- Convolutional Layers: This is where the model "learns" and extracts important features from the image.

- ReLU Activation: The ReLU function adds non-linearity to the model, helping it recognize more complex patterns.

MaxPooling: This process efficiently reduces image dimensions while keeping key features.

Fully Connected Layers: After features are extracted, the image data goes to these layers for the final classification.

Softmax Output: The last layer uses the Softmax function to give you probabilities (likelihoods) for each hair style type (e.g., "Wavy" 90%, "Straight" 5%, etc.).

Output and Recommendations
Once the model finishes its work, HairNet will give you these outputs:

Hair Style Prediction: You'll see a clear label showing the predicted hair type (e.g., "Wavy," "Straight," "Curly," "Dreadlocks," "Kinky").

Accuracy Display: The system will also show how accurate the model is based on the training and testing data.

Result Visualization: The image you put in will be displayed again, along with its predicted label. So, you can see the results immediately.

Hair Care Recommendations: And here's the cool part! The system will also give you personalized hair care tips that match the identified hair type.

The process then finishes, ready to be repeated for new images you want to process.

Dataset Used
For this project, I used a publicly available dataset from Kaggle:

Hair Type Dataset: https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset

Project Files
My project includes two main Jupyter Notebook (.ipynb) files:

FaridMh_4TIE_HairnetCNN.ipynb:

This notebook contains everything related to the core Deep Learning process.

It covers how to load and preprocess the data, building the CNN model architecture, the training process, and evaluating its performance.

HairnetCNNDEMO_FaridMuhammadHidayat_4TIE.ipynb:

This notebook is specifically for demonstrating the HairNet system via a web interface.

It includes the code for loading the trained model and features a temporary deployment using Gradio, allowing users to interact with the system through a simple web browser.


![image](https://github.com/user-attachments/assets/81c48df5-2394-4d7f-ab6b-40118ae226cb)

![image](https://github.com/user-attachments/assets/f57473ca-3084-4137-bd31-64d699303a2c)


