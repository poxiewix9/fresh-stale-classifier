# Fresh vs Stale Classifier

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A deep learning project using **MobileNetV2** to classify images of fruits and vegetables as either **fresh** or **stale**. This repository includes Python scripts to process data, train the model, and deploy it as a FastAPI service.

## Final Results

After training, the model achieved:
- **Test Accuracy:** 94.15%
- **Test Loss:** 0.146

## Project Structure

```
fresh-stale-api/
├── app.py                 # FastAPI application for classification
├── train_model.py         # Training script
├── train_from_kaggle.py   # Training from Kaggle dataset
├── test_local.py          # Local testing script
├── requirements.txt       # Python dependencies
├── best_model.h5          # Trained model weights
├── src/                   # Source code utilities
│   ├── prepare_data.py    # Data preparation
│   ├── train.py           # Training utilities
│   ├── evaluate.py        # Model evaluation
│   └── utils.py           # Helper functions
└── data/                  # Dataset directory (not included in repo)
```

## Getting Started

### Prerequisites

- Python **3.10+**
- Access to the [Kaggle dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification/data)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/poxiewix9/fresh-stale-classifier.git
cd fresh-stale-classifier
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

### Option 1: Train from Kaggle Dataset

1. Set up Kaggle API credentials:
```bash
# Save kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

2. Run the training script:
```bash
python train_from_kaggle.py
```

This will automatically:
- Download the dataset from Kaggle
- Prepare and split the data
- Train the model
- Save the best model as `best_model.h5`

### Option 2: Train with Local Data

1. Prepare your dataset in `data/raw/dataset/Train/` with folders like:
   - `freshapples/`
   - `rottenapples/`
   - etc.

2. Run the training script:
```bash
python train_model.py --data_dir data/raw/dataset_split --epochs 12
```

## Running the API

### Local Development

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Health check
- `POST /classify` - Classify an image as fresh or stale
  - Body: `{ "imageUrl": "https://..." }`
  - Response: `{ "isFresh": true, "confidence": 0.95, "model": "fresh-stale-classifier" }`

### Testing Locally

```bash
python test_local.py
```

## Deployment

### Railway

The repository is configured for Railway deployment. Simply connect your GitHub repository to Railway and it will automatically deploy.

Configuration files:
- `railway.json` - Railway configuration
- `nixpacks.toml` - Build configuration

### Other Platforms

- **Render**: Use `render.yaml`
- **Custom**: Deploy `app.py` as a FastAPI application

## License

This project is licensed under the MIT License.
