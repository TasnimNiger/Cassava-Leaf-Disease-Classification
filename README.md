# Cassava-Leaf-Disease-Classification
### 1. Introduction
Cassava is a staple food crop in many tropical regions, and its health is critical for food security. Cassava plants can suffer from various diseases that affect their leaves, leading to reduced yields. This project aims to classify cassava leaves into different disease categories or identify them as healthy using deep learning models. I employed the deep learning model Convolutional Neural Networks (CNN); if computational resources are limited, we can consider MobileNetV2, ResNet50, VGG16, EfficientNetB0/B1/B2, and InceptionNet pre-trained model.

### 2. Dataset
The dataset used for this project is the Cassava Leaf Disease Classification dataset from Kaggle. It consists of images of cassava leaves categorized into five classes:

-  Cassava Bacterial Blight (CBB)
-  Cassava Brown Streak Disease (CBSD)
-  Cassava Green Mottle (CGM)
-  Cassava Mosaic Disease (CMD)
-  Healthy

The dataset is divided into separate directories for training and testing, each containing subdirectories for each class.

### 3. Exploratory Data Analysis (EDA)
#### 3.1 Distribution of Classes
The class distribution in the training set was examined to check for any imbalance. It was found that the dataset had an imbalanced distribution across the different classes, which could affect the model performance.

**Distribution of Classes in the Training Set**
![image](https://github.com/user-attachments/assets/e8419296-6ed3-41ef-ad2c-270f80e42f61)

**Distribution of Classes in Test Set**
![image](https://github.com/user-attachments/assets/0a68d35b-f291-4913-a9a2-c4b692e6c7f5)

#### 3.2 Data Insights
Sample images from each class were visually inspected to assess image quality and variability. The images varied in resolution, lighting conditions, and angles, presenting a challenge for model training.
![image](https://github.com/user-attachments/assets/622cb13e-8803-45ca-8213-cb8d306c0e4b)

After performing EDA as outlined in the previous steps. The key findings are:

- **Class Distribution:** The dataset is imbalanced, with certain classes having more images than others.

-  **Image Quality:** The images vary in resolution and quality.

-  **Anomalies:** Some images have significantly different resolutions.

### 4. Data Preprocessing

#### 4.1 Image Resizing and Normalization
All images were standardized to a fixed size of 224x224 pixels and normalized to have pixel values with a mean of [0.485, 0.456, 0.406] and a standard deviation of [0.229, 0.224, 0.225].

#### 4.2 Augmentation
Data augmentation techniques such as random horizontal and vertical flips, random rotations, and color jitter were applied to increase dataset diversity and reduce overfitting.

### 5. Model Engineering

#### 5.1 Dataset Splitting
The dataset was split into training, validation, and test sets with an 85-15 split for training and validation.

#### 5.2 Model Architectures

**Convolutional Neural Network (CNN):**
A simple CNN was designed with three convolutional layers followed by max-pooling layers, and two fully connected layers. Dropout was used to prevent overfitting.

### 6. Model Training and Validation
Each model was trained for 25 epochs using the Adam optimizer and a learning rate of 0.001. The training process involved monitoring the training and validation loss and accuracy at each epoch.

### 7. Evaluation and Analysis
#### 7.1  Model Performance
The performance of each model was evaluated on the test set. The evaluation metrics included accuracy, precision, recall, and F1-score for each class. Additionally, the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) were plotted for each model to assess their performance.

#### 7.2 Result Analysis
Model	Test Accuracy (%)
CNN	0.72

Detailed classification reports were generated for the model, showing the precision, recall, and F1-score for each class.

![image](https://github.com/user-attachments/assets/4199dbe0-ac11-4aaf-983b-2c1c0d6a6c84)

#### 7.3 ROC Curve and AUC
Additionally, the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) were plotted for each model to assess their performance.

![image](https://github.com/user-attachments/assets/a2145451-1249-4852-a3b9-91459bf794cf)

#### 7.4 Sample Predictions
Sample images from the test set were displayed with their true and predicted labels to inspect the model performance visually. This helped in understanding the types of errors made by the models.

![image](https://github.com/user-attachments/assets/ddc89e0e-3334-48ff-92c4-827f4ece9950)

### 8. Conclusion
This structured approach, combining EDA, preprocessing, model engineering, and evaluation, provides a solid foundation for tackling the Cassava Leaf Disease Classification problem effectively. Here in this, CNN model showing accuracy 0.72 percent though transformer model may provide higher accuracy. Such as, I trained the with model MobileNetv2 and got highest accuracy 80.96 percent.

### References
[1] Cassava Leaf Disease Classification Dataset https://www.kaggle.com/datasets/gauravduttakiit/cassava-leaf-disease-classification

