A Multi-dimensional Information Restoration Strategy: Synergistic Modeling and Application of MultKAN in Wastewater Quality Prediction

License: MIT

Description

This repository contains the official source code for the paper "A Multi-dimensional Information Restoration Strategy: Synergistic Modeling and Application of MultKAN in Wastewater Quality Prediction".

This project proposes a novel model named MultKAN for the precise prediction of wastewater quality. The repository includes the complete source code for the MultKAN model, implementations of several baseline models used for comparison (e.g., LSTM, RandomForest, XGBoost), and the detailed experimental workflow.

Dataset

Important Note: Due to requirements from the wastewater treatment plant and related policies, the original dataset used in this research cannot be made publicly available.

This code repository does not contain any training or testing data. Users are required to prepare their own datasets to run and evaluate the models. To ensure model performance, we recommend a dataset size of over 1000 samples.

Environment Requirements

The code in this project has been tested on a system with the following configuration. To ensure reproducibility, it is recommended to use a similar or identical hardware and software environment.

Hardware Configuration

CPU: 2 x Intel® Xeon® Platinum 8575C Processor

GPU: 1 x NVIDIA RTX 4090 D

RAM: 128 GB or higher is recommended

Software Configuration

Operating System (OS): Windows 11

CUDA Version: 11.7

Python Version: 3.9

Python Libraries
# --- Core Data Science & Plotting ---
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2

# --- Machine Learning & Scikit-learn ---
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.0.0
catboost==1.2

# --- Deep Learning: PyTorch & TensorFlow (for CUDA 11.7) ---
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
tensorflow==2.11.0
keras==2.11.0

# --- Signal Processing ---
PyEMD==1.0.0

Installation

It is highly recommended to use a virtual environment (e.g., Conda) to manage project dependencies and avoid conflicts.

# Create a new environment named 'multkan_env' with a specific Python version
conda create -n multkan_env python=3.9

# Activate the newly created environment
conda activate multkan_env

Install TensorFlow (for CUDA 11.7):

