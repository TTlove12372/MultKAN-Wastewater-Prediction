# A Multi-dimensional Information Restoration Strategy: Synergistic Modeling and Application of MultKAN in Wastewater Quality Prediction


📝 Description
This repository contains the official source code for the paper "A Multi-dimensional Information Restoration Strategy: Synergistic Modeling and Application of MultKAN in Wastewater Quality Prediction".

This project proposes a novel model named MultKAN for the precise prediction of wastewater quality. The repository includes the complete source code for the MultKAN model, implementations of several baseline models used for comparison (e.g., LSTM, RandomForest, XGBoost), and the detailed experimental workflow.

# 📁 Dataset
Important Note: Due to requirements from the wastewater treatment plant and related policies, the original dataset used in this research cannot be made publicly available.

This code repository does not contain any training or testing data. Users are required to prepare their own datasets to run and evaluate the models. To ensure model performance, we recommend a dataset size of over 1000 samples.

# ⚙️ Environment Requirements
The code in this project has been tested on a system with the following configuration. To ensure reproducibility, it is recommended to use a similar or identical hardware and software environment.

Hardware Configuration
CPU: 2 x Intel® Xeon® Platinum 8575C Processor

GPU: 1 x NVIDIA RTX 4090 D

RAM: 128 GB or higher is recommended


# Export to Sheets
🔧 Installation
It is highly recommended to use a virtual environment (e.g., Conda) to manage project dependencies.

Create and Activate Conda Virtual Environment

📋 conda create -n multkan_env python=3.9

📋 conda activate multkan_env

Install PyTorch (for CUDA 11.7)

📋 pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

Install TensorFlow (for CUDA 11.7)

📋 pip install tensorflow==2.11.0

Install Remaining Dependencies

📋 pip install -r requirements.txt


📜 License
This project is licensed under the MIT License. For details, please see the LICENSE file.
