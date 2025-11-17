# Fresh vs Stale Classifier

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A deep learning project using **MobileNetV2** to classify images of fruits and vegetables as either **fresh** or **stale**. This repository includes Python scripts to process data, train the model, and evaluate its performance.

---

## Table of Contents

- [Final Results](#final-results)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
- [Running the Pipeline](#running-the-pipeline)  
- [License](#license)  

---

## Final Results

After **8 epochs** of training, the model achieved:

- **Test Accuracy:** 94.15%  
- **Test Loss:** 0.146  

Training history: *(You can add `logs/training_history.png` to the repository to visualize training progress.)*

---

## Project Structure
fresh-stale-classifier/
├─ data/
│ └─ raw/dataset/Train/ # Place your raw dataset here
├─ logs/ # Training logs and history
├─ models/
│ └─ fresh_stale_model/ # Saved model files
├─ src/
│ ├─ prepare_data.py # Prepares and splits dataset
│ ├─ train.py # Trains the model
│ └─ evaluate.py # Evaluates the trained model
├─ requirements.txt
└─ README.md


---

## Getting Started

### Prerequisites

- Python **3.10+**  
- Access to the dataset (see below)  

### Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd fresh-stale-classifier
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Prepare the Dataset

This project does not include the dataset due to size constraints.

Download the dataset (e.g., from Kaggle).

Unzip the files.

Organize the training images (e.g., freshapples, rottenapples, etc.) into:

data/raw/dataset/Train
```

Running the Pipeline
Step 1: Prepare and Split the Data

This script will automatically create train, validation, and test splits:
python src/prepare_data.py

Step 2: Train the Model

Train the model and save the best version as best_model.h5:
python src/train.py

Step 3: Evaluate the Model

Load the saved model and print the final classification report:
python src/evaluate.py

License
This project is licensed under the MIT License.

