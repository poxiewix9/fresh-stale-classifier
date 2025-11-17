import argparse
import os
from pathlib import Path
import shutil
from utils import split_dataset, ensure_dir #
import sys

def main():
    parser = argparse.ArgumentParser()


    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent


    DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Train"
    DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_split"


    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR) # Use new robust default
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR) # Use new robust default
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    if not raw_path.exists():
        print(f"Error: Source directory does not exist: {args.raw_dir}")
        print("Please check the path. It should be your 'Train' folder.")
        sys.exit(1)

    if Path(args.out_dir).exists():
        print(f"Removing old split directory: {args.out_dir}")
        shutil.rmtree(args.out_dir)

    print(f"Splitting data from {args.raw_dir} into {args.out_dir}...")
    split_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=42
    )
    print("Data splitting complete.")
    print(f"Your new data is ready in {args.out_dir}")
    print("\n" + "="*30)
    print("IMPORTANT: Run training with this new directory:")
    print(f"python src/train.py --data_dir {Path(args.out_dir).relative_to(PROJECT_ROOT)}")
    print("="*30)

if __name__ == "__main__":
    main()