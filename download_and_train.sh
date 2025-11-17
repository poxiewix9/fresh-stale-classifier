#!/bin/bash

# Download Kaggle dataset and train model
set -e

echo "🍎 Fresh/Stale Model Training"
echo "=============================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle API..."
    pip install kaggle
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle API credentials not found!"
    echo ""
    echo "Please:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Click 'Create New API Token'"
    echo "3. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "📥 Downloading dataset from Kaggle..."
cd "$(dirname "$0")"
mkdir -p data/raw
cd data/raw

# Download dataset
kaggle datasets download -d swoyam2609/fresh-and-stale-classification -p .

echo ""
echo "📦 Extracting dataset..."
unzip -q fresh-and-stale-classification.zip -d dataset 2>/dev/null || unzip -q *.zip -d dataset

echo ""
echo "📁 Checking dataset structure..."
find dataset -type d -maxdepth 2 | head -10

echo ""
echo "🔄 Preparing data splits..."
cd ../..
python src/prepare_data.py --raw_dir data/raw/dataset/Train --out_dir data/raw/dataset_split

echo ""
echo "🚀 Training model..."
python train_model.py --data_dir data/raw/dataset_split --epochs 12

echo ""
echo "✅ Training complete! Model saved as best_model.h5"
echo ""
echo "Next steps:"
echo "1. git add best_model.h5"
echo "2. git commit -m 'Retrained model'"
echo "3. git push"
echo "4. Railway will auto-deploy!"

