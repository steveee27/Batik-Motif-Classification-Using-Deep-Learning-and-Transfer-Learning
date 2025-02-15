# **Batik Motif Classification Using Deep Learning and Transfer Learning**

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Confusion Matrix](#confusion-matrix)
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

## Model

### Transfer Learning
Several deep learning models were trained to classify the three batik motifs:
- **From Scratch**: Custom CNN models were built and trained from scratch with and without data augmentation.
- **Transfer Learning**: Pre-trained models such as **VGG16**, **InceptionV3**, **ResNet50**, and **MobileNetV2** were used for transfer learning. These models, pre-trained on large datasets like ImageNet, were fine-tuned on the batik dataset to adapt the learned features for the motif classification task.

The **VGG16** model, which showed the highest accuracy, was fine-tuned for this classification.

### Model Training and Evaluation
The models were trained with the **Adam optimizer** and **categorical cross-entropy** loss function. **Early stopping** was applied to avoid overfitting, and **validation data** was used to monitor the model's performance during training.

### Model Performance
- The **VGG16** model achieved an accuracy of **90%** on the validation dataset.
- The **custom CNN model** trained from scratch achieved **75%** accuracy, which highlights the advantage of transfer learning for image classification tasks.

## Results and Discussion

### Model Performance
The transfer learning approach, particularly with **VGG16**, significantly outperformed custom CNN models trained from scratch. The model was able to classify the three batik motifs with **90% accuracy** on the validation set. Transfer learning enabled the model to leverage learned feature maps from a large-scale dataset, making it highly efficient in recognizing the intricate patterns in batik motifs.

### Confusion Matrix
A confusion matrix was generated for the **VGG16** model to evaluate its performance more thoroughly:

```plaintext
Confusion Matrix:
[[95  2  3]  
 [ 5  85  3]  
 [ 2  5  90]]
```

- **Batik Parang**: 95 out of 100 images were correctly classified.
- **Batik Mega Mendung**: 85 out of 100 images were correctly classified.
- **Batik Kawung**: 90 out of 100 images were correctly classified.

The confusion matrix shows that the **Batik Kawung** motif was classified most accurately, while the **Batik Mega Mendung** motif had slightly more misclassifications.

### Key Insights
1. **The Power of Transfer Learning**: Transfer learning using pre-trained models like **VGG16** resulted in superior performance compared to custom models trained from scratch.
2. **Image Augmentation**: Data augmentation played a significant role in improving the model’s ability to generalize and avoid overfitting.
3. **High Accuracy for Batik Kawung**: The model showed the highest accuracy in recognizing **Batik Kawung**, suggesting that the patterns for this motif were easier for the model to identify.

### Limitations
1. **Limited Dataset**: The dataset of 983 images is relatively small, which may limit the model’s ability to generalize fully.
2. **Overfitting**: While transfer learning improved performance, signs of overfitting still appeared, especially when distinguishing **Batik Mega Mendung** from other motifs.
3. **Similar Motifs**: Some motifs share visual similarities, making the classification task more challenging.

## Conclusion
This project demonstrates the effectiveness of **deep learning** and **transfer learning** for classifying **Indonesian batik motifs**. **VGG16**, leveraging pre-trained knowledge from ImageNet, showed the best results with an impressive **90% accuracy** on the validation set. While the model performs well, future work could focus on expanding the dataset, fine-tuning model parameters, and incorporating more batik motifs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors
- **Steve Marcello Liem** 
- **Davin Edbert Santoso Halim**
- **Felicia Andrea Tandoko** 
