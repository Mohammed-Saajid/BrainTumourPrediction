# Brain Tumor Prediction using ResNet Architecture

This repository contains a deep learning project for predicting brain tumors using CT and MRI images. The model leverages a custom-designed ResNet architecture implemented using TensorFlow Core API. The training pipeline includes a custom training loop and distributed GPU training for efficiency and scalability.

## Key Features

1. **ResNet Architecture**: The model architecture is based on the ResNet (Residual Network) design, which is well-suited for extracting complex patterns from medical imaging data.

2. **TensorFlow Core API**: The architecture is built from scratch using TensorFlow Core API for maximum control and flexibility.

3. **Custom Training Loop**: A custom training loop was implemented to provide fine-grained control over the training process, loss calculation, and metrics tracking.

4. **Distributed GPU Training**: To handle large datasets and expedite training, distributed training across multiple GPUs was employed using TensorFlow's distribution strategies.

## Dataset

The dataset consists of labeled CT and MRI brain images categorized into two classes:

1. Tumor

2. Non-Tumor

## Preprocessing Steps:

1. Resized all images to a fixed size of 224x224 pixels.

2. Normalized pixel values to the range [0, 1].


## Model Architecture

The ResNet architecture used in this project comprises the following components:

1. Input Layer: Accepts 224x224x3 image tensors.

2. Residual Blocks: Stacked blocks of convolutional layers with skip connections to alleviate the vanishing gradient problem.

3. Global Average Pooling: Reduces the spatial dimensions while retaining key features.

4. Dense Layers: Fully connected layers with ReLU activation.

5. Output Layer: Single neuron with sigmoid activation for binary classification.

## Implementation Details

**Dependencies**

- TensorFlow 

- NumPy

- Matplotlib

- scikit-learn

## Training Pipeline

- **Model Definition**: The ResNet architecture was defined using TensorFlow Core API.

- **Loss Function**: Binary cross-entropy loss was used to quantify the difference between predicted and actual labels.

- **Optimizer**: Adam optimizer with a learning rate scheduler for dynamic learning rate adjustments.

- **Metrics**: Accuracy were tracked during training.

- **Custom Training Loop**: Included forward propagation, backpropagation, and metrics logging for each batch.

- **Distributed Training**: TensorFlow's tf.distribute.MirroredStrategy was used to enable multi-GPU training.

## Training Results

- Training Accuracy: Achieved >90% accuracy after 80 epochs.

- Validation Accuracy: Achieved > 90%.




## Acknowledgments

1. Open-source frameworks like TensorFlow and NumPy.

2. Publicly available datasets for medical imaging research.
   Dataset Link:- [DataSet](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)

3. Community support and resources that facilitated the development of this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



