# Diabetes-Prediction
A deep learning-based system to predict diabetes using health data. It involves data cleaning, preprocessing, and training an artificial neural network on features like glucose, BMI, age, and insulin levels. The model helps in early detection and supports healthcare decision-making.
 Diabetes Prediction using Deep Learning

This project focuses on building a deep learning model to predict the presence of diabetes in patients using medical diagnostic data. The dataset used is the Pima Indians Diabetes Database, a standard dataset for binary classification in the medical domain.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [How to Run](#how-to-run)
- [License](#license)

## Overview

This notebook implements a deep learning approach using TensorFlow and Keras to classify whether a person is diabetic based on features like glucose level, BMI, age, etc. The main goals are:

- Preprocessing and normalization of data
- Designing and training a neural network
- Evaluating performance using metrics like accuracy and loss

## Dataset

The dataset used is the **Pima Indians Diabetes Database**, which includes the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target variable)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- TensorFlow / Keras
- Scikit-learn

## Model Architecture

The model is a feed-forward neural network with the following structure:

- Input Layer: 8 features
- Hidden Layers: Two Dense layers with ReLU activation
- Output Layer: One neuron with Sigmoid activation (binary classification)

Optimizer: `adam`  
Loss Function: `binary_crossentropy`  
Evaluation Metric: `accuracy`

## Results

The model achieves a reasonable accuracy after training, showing its effectiveness in binary classification for medical diagnosis. Training and validation loss/accuracy curves are also plotted to visualize performance.

## How to Run

1. Clone the repository or download the notebook.
2. Make sure you have Python and the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```
3. Run the notebook `Diabetes_deep_learning.ipynb` step-by-step in a Jupyter environment.

## License

This project is open-source and available under the MIT License.
