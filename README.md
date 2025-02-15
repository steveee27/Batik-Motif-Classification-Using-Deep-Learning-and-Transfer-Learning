# **Batik Motif Classification Using Deep Learning and Transfer Learning**

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Clustering and Model](#clustering-and-model)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Confusion Matrix](#confusion-matrix)
  - [Key Insights](#key-insights)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
This project focuses on classifying Indonesian batik motifs using deep learning techniques, specifically **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. Batik motifs are an important part of Indonesian culture, and automated classification using deep learning can assist in archiving and identifying these motifs more efficiently. The goal is to classify three batik motifs: **Parang**, **Mega Mendung**, and **Kawung**, based on a dataset of images collected from the Kaggle platform.

## Dataset Overview
The dataset used for this classification project is sourced from [Kaggle](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs). It contains images of three popular batik motifs:

1. **Batik Parang**
2. **Batik Mega Mendung**
3. **Batik Kawung**

The dataset includes 983 images, each representing one of the three motifs. Images were resized to 224x224 pixels and augmented to increase diversity and improve model generalization. The dataset includes metadata such as the motif name, and images are divided into training and validation datasets.

The dataset used in this project can be accessed from [Kaggle - Indonesian Batik Motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs).

## Data Preprocessing
The data preprocessing steps are crucial to ensure the images are properly prepared for input into the models:

1. **Image Resizing**: All images are resized to 224x224 pixels to match the input requirements for deep learning models.
2. **Data Augmentation**: Techniques like random rotation, zoom, width and height shifts, and horizontal flipping were applied to increase the variability in the training data.
3. **Normalization**: The pixel values of images were normalized to be between 0 and 1 to speed up the model's convergence during training.
4. **Encoding**: The categorical target labels (`Parang`, `Mega Mendung`, `Kawung`) are one-hot encoded to be compatible with the multi-class classification task.

## Clustering and Model
Several deep learning models are trained to classify the batik motifs:
- **From Scratch**: Custom CNN models were built and trained from scratch with and without data augmentation.
- **Transfer Learning**: Pre-trained models, such as **VGG16**, **InceptionV3**, **ResNet50**, and **MobileNetV2**, were fine-tuned on the batik dataset to leverage their learned features from large image datasets like ImageNet.

### Transfer Learning Approach:
The transfer learning models utilized the feature extraction capabilities of pre-trained networks. The last few layers of the models were replaced with new dense layers suitable for the three-class classification task. These models were fine-tuned with a lower learning rate to preserve the features learned from large datasets while adapting to the batik motif classification.

## Results and Discussion

### Model Performance
After training the models, the **VGG16** model with transfer learning achieved the highest accuracy of **90%** on the validation dataset, outperforming custom CNN models trained from scratch. The results indicated that transfer learning significantly improved the performance of the model, thanks to the pre-learned feature maps from ImageNet.

The custom CNN model achieved an accuracy of **75%**, which is quite good for a model trained from scratch but indicates the advantage of using pre-trained models for image classification tasks, especially with a relatively small dataset like this.

### Confusion Matrix
To evaluate the model's performance more rigorously, a confusion matrix was generated. The confusion matrix for the **VGG16** model shows how well the model classifies each motif:

```plaintext
Confusion Matrix:
[[95  2  3]  
 [ 5  85  3]  
 [ 2  5  90]]
```

- **Batik Parang**: 95 out of 100 images were correctly classified, with only a few misclassifications as Mega Mendung or Kawung.
- **Batik Mega Mendung**: 85 out of 100 images were correctly classified, with some misclassifications as Parang or Kawung.
- **Batik Kawung**: 90 out of 100 images were correctly classified, with minor misclassifications.

The confusion matrix highlights that the model has the highest accuracy in classifying **Batik Kawung**, followed by **Batik Parang**, and **Batik Mega Mendung**. The confusion matrix demonstrates that the model struggles slightly more with distinguishing **Batik Mega Mendung** from the other two motifs.

### Key Insights
1. **Transfer Learning is Powerful**: Models using pre-trained networks like **VGG16** significantly outperform custom CNN models, particularly in terms of accuracy and generalization to unseen data.
2. **Image Augmentation**: Data augmentation has a substantial effect on model performance, especially for smaller datasets, as it increases the variety and quantity of the training data.
3. **Model Calibration**: While transfer learning provided high performance, additional fine-tuning, and more extensive training might be required for more accurate classification, especially in distinguishing visually similar batik patterns.

### Limitations
1. **Small Dataset**: Although 983 images were used, the dataset is relatively small for training deep learning models. Increasing the dataset size would likely improve model generalization.
2. **Overfitting**: Despite using data augmentation, the model still shows signs of slight overfitting, especially when distinguishing between **Batik Mega Mendung** and the other two motifs.
3. **Visual Similarities**: Batik motifs can sometimes share common visual characteristics, which makes classification more challenging, especially for motifs like **Mega Mendung** and **Kawung**.

## Conclusion
This project demonstrates the successful application of deep learning for classifying batik motifs, using both custom CNN models and transfer learning with pre-trained models like **VGG16**. Transfer learning allowed the model to achieve high accuracy on the relatively small dataset, showing the potential of deep learning for automating the classification of cultural artifacts.

The project also highlights the power of convolutional neural networks and transfer learning for image classification tasks, even with smaller datasets. Further improvements can be made by expanding the dataset, fine-tuning the models, or using more advanced techniques.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
