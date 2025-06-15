# Human Activity Recognition (HAR) Using CNN + LSTM

A deep learning project that combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks to classify physical human activities using wearable sensor data. The model is deployed via a user-friendly **Streamlit** web application.

## Table of Contents

- [Features](#features)
- [Model](#model)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Model (Jupyter Notebook)](#1-train-the-model-jupyter-notebook)
  - [2. Run Streamlit App](#2-run-streamlit-app)
  - [3. Upload Test CSV](#3-upload-test-csv)
- [Project Structure](#project-structure)
- [Model Training](#model-training)

## Features

- CNN extracts local temporal features from time-series sensor data.
- LSTM captures sequential (time-dependent) patterns.
- Real-time prediction via Streamlit web interface.
- Supports CSV uploads for batch predictions.
- Downloadable results in CSV format.

## Model

- Architecture: `Conv1D → MaxPooling1D → Dropout → LSTM → Dense → Softmax`
- Trained on the **UCI HAR Dataset** with 561 features per sample.
- Achieved **~88% test accuracy**.

## Prerequisites

- Python 3.7 or higher
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/har-cnn-lstm-streamlit.git
   cd har-cnn-lstm-streamlit

## Install Dependencies
`pip install -r requirements.txt`

## Usage
## 1. Train the Model (Jupyter Notebook)
Open the notebook:
  `jupyter notebook har-project.ipynb`
Train the model and export the .h5 file.

## 2. Run Streamlit App
`streamlit run app.py`
This opens the web app where you can upload test CSV files.

## 3. Upload Test CSV
Format: 561 features per row.
Model returns activity predictions like WALKING, SITTING, STANDING, etc.
You can download the result as a CSV file.

## Model Training

The model training and evaluation are done inside `har-project.ipynb`.  
It includes:
- Loading and reshaping sensor data
- Label encoding
- Model architecture (CNN + LSTM)
- Training performance
- Confusion matrix and classification report

## Project Structure
```har-cnn-lstm-streamlit/
├── app.py                 # Streamlit web app
├── har-project.ipynb      # Training notebook
├── har_cnn_lstm_model.h5  # Trained model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation```



