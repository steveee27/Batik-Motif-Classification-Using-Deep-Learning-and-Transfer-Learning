# **Batik Motif Classification Using Deep Learning and Transfer Learning**

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)
- [Contributors](#contributors)

## Introduction
This project focuses on classifying three popular Indonesian batik motifs—**Parang**, **Mega Mendung**, and **Kawung**—using deep learning techniques, specifically **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. These motifs are well-known in Indonesian culture, and this automated classification system aims to assist in identifying these motifs efficiently.

While the dataset contains a variety of batik motifs, this project specifically focuses on **Batik Parang**, **Batik Mega Mendung**, and **Batik Kawung**, leveraging the power of deep learning to classify these motifs with high accuracy.

## Dataset Overview
The dataset for this classification project is sourced from [Kaggle - Indonesian Batik Motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs), which includes various batik motifs. For this project, we focus on three iconic motifs:

1. **Batik Parang**: 50 images
2. **Batik Mega Mendung**: 46 images
3. **Batik Kawung**: 45 images

While the full dataset contains **983 images**, this project narrows the focus to these three motifs to demonstrate deep learning techniques in image classification. 

## Methodology
![image](https://github.com/user-attachments/assets/1f113663-9c44-406d-83ef-ce0143f3373b)

## Data Preprocessing
The data preprocessing steps are essential to ensure the images are properly prepared for deep learning models:
1. **Image Resizing**: All images were resized to **224x224 pixels**, as required by most deep learning architectures.
2. **Data Augmentation**: Techniques such as rotation, zooming, and horizontal flipping were applied to increase the diversity of the dataset, preventing overfitting.
3. **Normalization**: Pixel values were normalized to range between 0 and 1.
4. **Encoding**: The target labels (`Parang`, `Mega Mendung`, and `Kawung`) were one-hot encoded for multi-class classification.

---

### **Model**

#### **Transfer Learning**

Several deep learning models were trained to classify the three batik motifs:

- **From Scratch**: Custom CNN models were built and trained from scratch with and without data augmentation.
- **Transfer Learning**: Pre-trained models such as **VGG16**, **InceptionV3**, **ResNet50**, and **MobileNetV2** were used for transfer learning. These models, pre-trained on large datasets like ImageNet, were fine-tuned on the batik dataset to adapt the learned features for motif classification tasks.

Among these, **MobileNetV2** achieved the highest accuracy and was the model of choice for this classification.

#### **Model Training and Evaluation**

The models were trained using the **Adam optimizer** and **categorical cross-entropy** loss function. Early stopping was applied to prevent overfitting, and validation data was used throughout the training to monitor the model's performance.

---

### **Results and Discussion**

#### **Model Performance**

The transfer learning approach, particularly using **MobileNetV2**, significantly outperformed custom CNN models trained from scratch. MobileNetV2, a highly efficient model, was fine-tuned to classify the three batik motifs, achieving **93.75% accuracy** on the validation set. This demonstrates the power of transfer learning in handling smaller, specialized datasets.

#### **Key Insights**

1. **The Power of Transfer Learning**: Transfer learning, especially using **MobileNetV2**, provided the best performance compared to custom models trained from scratch.
2. **Image Augmentation**: Data augmentation was crucial in preventing overfitting, allowing the model to generalize better to unseen data.
3. **High Accuracy for Batik Kawung**: The model showed the highest accuracy in recognizing **Batik Kawung**, indicating that the visual patterns of this motif were clearer for the model to distinguish.

#### **Limitations**

1. **Limited Dataset**: The relatively small dataset of 983 images may have limited the model's ability to generalize fully.
2. **Overfitting**: Despite using transfer learning, some overfitting signs appeared, particularly when distinguishing **Batik Mega Mendung** from the other motifs.
3. **Similar Motifs**: Certain motifs share visual features that made classification more challenging, especially in differentiating between **Batik Mega Mendung** and **Batik Parang**.

## **Conclusion**

This project demonstrates the effectiveness of **deep learning** and **transfer learning** for classifying **Indonesian Batik motifs**. **MobileNetV2** was the model of choice due to its high efficiency and **93.75% accuracy** on the validation set. The results show that transfer learning with pre-trained models provides significant improvements, especially when working with small datasets. Future work can focus on expanding the dataset, improving model fine-tuning, and incorporating additional batik motifs to enhance model accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors
- **Steve Marcello Liem** 
- **Davin Edbert Santoso Halim**
- **Felicia Andrea Tandoko** 
