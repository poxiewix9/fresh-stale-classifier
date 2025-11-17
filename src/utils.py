# src/utils.py
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def split_dataset(raw_dir, out_dir, val_frac=0.15, test_frac=0.15, seed=42):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found in {raw_dir}. Please ensure dataset has subfolders per class.")

    for cls in classes:
        cls_path = raw_dir / cls
        images = [p for p in cls_path.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
        train_and_val, test = train_test_split(images, test_size=test_frac, random_state=seed)
        train, val = train_test_split(train_and_val, test_size=val_frac/(1-test_frac), random_state=seed)

        for split_name, split_files in [('train', train), ('val', val), ('test', test)]:
            dest = out_dir / split_name / cls
            ensure_dir(dest)
            for src in tqdm(split_files, desc=f"{cls} -> {split_name}", leave=False):
                shutil.copy(src, dest / src.name)
