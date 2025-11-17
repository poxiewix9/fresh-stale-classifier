Fresh vs. Stale Fruit Classifier

This project uses a deep learning model (MobileNetV2) to classify images of fruits and vegetables as either 'fresh' or 'stale'.

This repository contains the Python source code to process the data, train the model, and evaluate its performance.

Final Results

After training for 8 epochs, the model achieved the following on the test set:

Test Accuracy: 94.15%

Test Loss: 0.146

Here is the training history:
(You can add the logs/training_history.png image to your GitHub repo manually after uploading to show your results)

How to Run This Project

1. Prerequisites

Python 3.10+

Access to the dataset (see below)

2. Setup

First, clone this repository and create your virtual environment:

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd fresh-stale-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


3. Get the Data

This project does not include the data due to its large size. You must download it yourself.

Download the dataset (e.g., from Kaggle or your source).

Unzip the files.

Place all the training images (the 18 folders like freshapples, rottenapples, etc.) into a single directory:
data/raw/dataset/Train

Your project structure should look like this before you run any scripts:

fresh-stale-classifier/
├── data/
│   └── raw/
│       └── dataset/
│           └── Train/
│               ├── freshapples/
│               ├── fresbanana/
│               ├── rottenapples/
│               └── ... (all 18 folders)
├── src/
│   ├── train.py
│   └── ...
└── requirements.txt


4. Run the Pipeline

Once your data is in the correct folder, follow these steps in your terminal:

Step 1: Prepare and split the data
This script will automatically create the train/, val/, and test/ splits.

python src/prepare_data.py


Step 2: Train the model
This will train the model, save the best one as best_model.h5, and create the models/fresh_stale_model directory.

python src/train.py


Step 3: Evaluate the model
This will load the saved model and print the final classification report.

python src/evaluate.py
