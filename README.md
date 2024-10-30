Fashion Recognition with YOLO and Fashion-MNIST

This project leverages the YOLO (You Only Look Once) model for image classification on the Fashion-MNIST dataset, a popular dataset for benchmarking machine learning algorithms in the fashion industry.
Overview

The goal of this project is to classify images of fashion items (e.g., t-shirts, pants, shoes) using YOLO for classification tasks, which extends YOLO's traditional use for object detection into the realm of image classification. We use YOLO's "Nano" configuration (yolo11n-cls.yaml) for an efficient, lightweight model suitable for Fashion-MNIST's 28x28 images.
Project Structure

This repository includes:

    train_model.py: Main script to train the YOLO model on Fashion-MNIST.
    test.py: Testing script to evaluate the model’s performance on new data.
    utils/: Folder containing any utility files or helper functions.
    datasets/: Directory to hold the Fashion-MNIST dataset.
    runs/: Directory where training results and model weights are saved automatically by YOLO.

Setup Instructions

    Install YOLO with Ultralytics
    First, ensure you have the YOLO library from Ultralytics installed:

    bash

pip install ultralytics

Dataset
Download or link the Fashion-MNIST dataset to the datasets/ directory. You can get the dataset from here if not already included.

Load and Train the Model
Open train_model.py and follow the code to load and train the YOLO model:

python

    from ultralytics import YOLO

    # Load YOLO model configuration
    model = YOLO("yolo11n-cls.yaml")

    # Train the model
    results = model.train(data="fashion-mnist", epochs=20, imgsz=28)

    This trains the YOLO model on Fashion-MNIST for 20 epochs with 28x28 image size, as these are Fashion-MNIST’s standard dimensions.

    Model Results
    After training, the model saves results, weights, logs, and other files in runs/train/exp/.

Key Files and Folders

    yolo11n-cls.yaml: YOLO model configuration file, defining architecture, hyperparameters, and layer settings.
    runs/train/exp/weights/best.pt: Best-performing model weights saved automatically for future inference or additional analysis.

Understanding YOLO for Classification

    YOLO Architecture for Classification
    YOLO traditionally performs object detection, but here it’s configured for classification with adjustments to the last layer to output class labels. Using a Nano model (11n), we maintain a lightweight setup suitable for low-dimensional images.

    Training Configurations
    The training process includes:
        Epochs: Set to 20 in this case.
        Image Size (imgsz): 28x28 pixels, aligning with Fashion-MNIST dimensions.

Why YOLO?

YOLO is known for its efficiency and versatility across multiple computer vision tasks. This project uses YOLO to:

    Simplify multi-task model handling (e.g., switch easily between classification and detection).
    Achieve fast and efficient training and inference, ideal for real-time applications.
    Provide flexible configurations via .yaml files to customize architectures.

Sample Training Code

python

from ultralytics import YOLO

# Initialize YOLO model with classification config
model = YOLO("yolo11n-cls.yaml")

# Train the model on Fashion-MNIST dataset
results = model.train(data="fashion-mnist", epochs=20, imgsz=28)

Results and Evaluation

    Weights: The best-performing weights are saved as best.pt in runs/train/exp/weights/, allowing easy re-use for inference.
    Training Logs: YOLO saves metrics, accuracy, and loss values to analyze model performance across epochs.
    Predictions: During training, prediction samples on the validation set are saved in runs/train/exp for quick visual assessments.

License

This project is licensed under the MIT License - see the LICENSE file for details.
