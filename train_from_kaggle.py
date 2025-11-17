"""
Download Kaggle dataset and train model
Run: python train_from_kaggle.py
"""
import os
import subprocess
import sys
from pathlib import Path

def check_kaggle():
    """Check if kaggle is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        print("Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return True

def download_dataset():
    """Download dataset from Kaggle"""
    print("📥 Downloading dataset from Kaggle...")
    
    # Check credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n⚠️  Kaggle API credentials not found!")
        print("\nPlease:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    
    # Download
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    os.chdir(data_dir)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'swoyam2609/fresh-and-stale-classification',
            path='.',
            unzip=True
        )
        print("✅ Dataset downloaded!")
        return True
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False
    finally:
        os.chdir("..")

def prepare_data():
    """Prepare data splits"""
    print("\n🔄 Preparing data splits...")
    try:
        from src.prepare_data import main as prepare_main
        import sys
        sys.argv = ['prepare_data.py', '--raw_dir', 'data/raw/dataset/Train', '--out_dir', 'data/raw/dataset_split']
        prepare_main()
        return True
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        print("Trying alternative method...")
        # Try manual split
        return False

def train_model():
    """Train the model"""
    print("\n🚀 Training model...")
    try:
        from train_model import main as train_main
        import sys
        sys.argv = ['train_model.py', '--data_dir', 'data/raw/dataset_split', '--epochs', '12']
        train_main()
        return True
    except Exception as e:
        print(f"❌ Error training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🍎 Fresh/Stale Model Training")
    print("=" * 40)
    
    if not check_kaggle():
        sys.exit(1)
    
    if not download_dataset():
        print("\n❌ Failed to download dataset")
        sys.exit(1)
    
    if not prepare_data():
        print("\n⚠️  Data preparation had issues, but continuing...")
    
    if train_model():
        print("\n✅ Training complete! Model saved as best_model.h5")
        print("\nNext steps:")
        print("1. git add best_model.h5")
        print("2. git commit -m 'Retrained model'")
        print("3. git push")
    else:
        print("\n❌ Training failed")
        sys.exit(1)

