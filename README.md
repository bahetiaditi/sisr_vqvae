# Super Resolution and Artifact Removal on Deep Lesion Dataset

## Overview
This project focuses on enhancing medical imaging by performing super-resolution on downsampled images and removing artifacts from bilinear interpolated images. Utilizing a deep learning approach, this project aims to reconstruct high-resolution images from their lower-resolution counterparts while simultaneously reducing noise and improving the overall quality.

## Results
The results showcase the ability of the model to enhance image resolution and remove artifacts. The following metrics have been used to quantify the performance:
- **Total Loss**: The combined measure of reconstruction error and any additional loss components, such as a vector quantization loss.
- **PSNR Score**: Peak Signal-to-Noise Ratio, a measure of how accurately the reconstructed image resembles the original high-resolution image.
- **SSIM Score**: Structural Similarity Index Measure, a perception-based metric that considers changes in structural information, luminance, and contrast.


## Dataset
The dataset comprises medical images sourced from the [Deep Lesion Dataset](https://nihcc.app.box.com/v/DeepLesion). Each image has undergone preprocessing steps, including resizing, normalization, and augmentation, to prepare for training.

## Model Architecture
The architecture is composed of several custom layers and blocks, including:
- **Vector Quantizer**: Quantizes the inputs to reduce the size of the representation.
- **Residual Blocks**: Facilitates the training of deeper architectures by learning residual functions.
- **Encoder-Decoder Structure**: Extracts features from images and reconstructs the output from these features.

## Training Process
The training involves several key steps:
1. Loading and preprocessing the dataset.
2. Splitting the dataset into training and test sets.
3. Training the model using the training set with periodic evaluation on the test set.
4. Computing loss and backpropagation to update the model weights.

## Evaluation and Visualization
The evaluation on the test set involves calculating the Total Loss, PSNR, and SSIM scores for each image. The visualization provides a side-by-side comparison of the original, downsampled, bilinear interpolated with artifacts, and reconstructed images to demonstrate the model's performance.

## Requirements
- Python 3.8 or later
- PyTorch 1.8.1 or later
- Matplotlib 3.4.1 or later
- scikit-image 0.18.1 or later

## Files and Directories
- `data_loader.py`: Contains the dataset class for loading and preprocessing.
- `models.py`: Defines the neural network models.
- `train.py`: Contains the training loop, including forward pass, loss computation, and backpropagation.
- `utils.py`: Utility functions for visualization and metric calculations.
- `main.py`: The main script that orchestrates the training and evaluation process.

## Conclusion
The project demonstrates a significant improvement in image quality with the enhanced resolution and artifact removal. These enhancements are critical in medical imaging, where the clarity of details can be crucial for diagnosis.


