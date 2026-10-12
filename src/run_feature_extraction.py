"""
run_feature_extraction.py
Equivalent of executing notebooks/02_ResNet50_Features.ipynb cell-by-cell.
Run from the SKIN_CANCER project root:
    python run_feature_extraction.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # src/ -> project root
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_CSV     = RESULTS_DIR / 'splits.csv'
LABEL_MAP_JSON = RESULTS_DIR / 'label_mapping.json'
FEATURES_NPZ   = RESULTS_DIR / 'resnet50_features.npz'

BATCH_SIZE  = 16
IMG_SIZE    = 224
LOG_EVERY   = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'PyTorch: {torch.__version__}')

# ── 1. Load & validate splits ────────────────────────────────────────────────
print('\n[1] Loading splits.csv...')
splits = pd.read_csv(SPLITS_CSV)
print(f'    Rows: {len(splits):,}  |  Columns: {list(splits.columns)}')
counts = splits['split'].value_counts()[['train', 'val', 'test']]
print(f'    train={counts["train"]:,}  val={counts["val"]:,}  test={counts["test"]:,}')

assert counts['train'] == 7054
assert counts['val']   == 1464
assert counts['test']  == 1497
print('    [OK] Split counts match expected.')

# ── 2. Label mapping ─────────────────────────────────────────────────────────
print('\n[2] Encoding labels...')
CLASSES = sorted(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
label_mapping = {cls: idx for idx, cls in enumerate(CLASSES)}
for k, v in label_mapping.items():
    print(f'    {v}  {k}')

with open(LABEL_MAP_JSON, 'w') as f:
    json.dump(label_mapping, f, indent=2)
print(f'    Saved: {LABEL_MAP_JSON}')

splits['label'] = splits['dx'].map(label_mapping)
assert splits['label'].isna().sum() == 0

# ── 3. Sanity-check per-class per-split counts ───────────────────────────────
print('\n[3] Per-class counts per split:')
pivot = (
    splits.groupby(['dx', 'split'])
    .size()
    .unstack(fill_value=0)
    [['train', 'val', 'test']]
)
pivot['total'] = pivot.sum(axis=1)
print(pivot.to_string())

# ── 4. Dataset / DataLoader ──────────────────────────────────────────────────
imagenet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths     = df['filepath'].tolist()
        self.labels    = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def make_loader(df_split):
    ds = SkinLesionDataset(df_split, transform=imagenet_transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=False)

df_train = splits[splits['split'] == 'train'].reset_index(drop=True)
df_val   = splits[splits['split'] == 'val'].reset_index(drop=True)
df_test  = splits[splits['split'] == 'test'].reset_index(drop=True)

train_loader = make_loader(df_train)
val_loader   = make_loader(df_val)
test_loader  = make_loader(df_test)

print(f'\n[4] DataLoaders ready')
print(f'    train: {len(df_train):,} images  ({len(train_loader)} batches)')
print(f'    val  : {len(df_val):,} images    ({len(val_loader)} batches)')
print(f'    test : {len(df_test):,} images    ({len(test_loader)} batches)')

# ── 5. Load ResNet50 backbone ────────────────────────────────────────────────
print('\n[5] Loading ResNet50 (ImageNet weights)...')
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()
total_params = sum(p.numel() for p in feature_extractor.parameters())
print(f'    Parameters : {total_params:,}')
print(f'    Output dim : 2048  (global avg-pool)')
print(f'    Mode       : eval / frozen')

# ── 6. Extraction function ───────────────────────────────────────────────────
def extract_features(loader, split_name):
    all_features, all_labels = [], []
    total_batches = len(loader)
    t0 = time.time()
    print(f'\n[EXTRACT:{split_name}] {len(loader.dataset):,} images, {total_batches} batches...')

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs  = imgs.to(device)
            feats = feature_extractor(imgs)
            feats = feats.view(feats.size(0), -1)
            all_features.append(feats.cpu().numpy())
            all_labels.append(np.array(labels))

            if (batch_idx + 1) % LOG_EVERY == 0 or (batch_idx + 1) == total_batches:
                imgs_done = min((batch_idx + 1) * BATCH_SIZE, len(loader.dataset))
                elapsed   = time.time() - t0
                print(f'  Batch {batch_idx+1:>4d}/{total_batches}'
                      f'  |  {imgs_done:>5,}/{len(loader.dataset):,} imgs'
                      f'  |  {elapsed:5.1f}s')

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    print(f'  => X.shape={X.shape}  y.shape={y.shape}  ({time.time()-t0:.1f}s total)')
    return X, y

# ── 7. Run extraction ────────────────────────────────────────────────────────
X_train, y_train = extract_features(train_loader, 'TRAIN')
X_val,   y_val   = extract_features(val_loader,   'VAL')
X_test,  y_test  = extract_features(test_loader,  'TEST')

# ── 8. Save .npz ────────────────────────────────────────────────────────────
print(f'\n[8] Saving features to {FEATURES_NPZ}...')
np.savez_compressed(
    FEATURES_NPZ,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
)
size_mb = FEATURES_NPZ.stat().st_size / (1024**2)
print(f'    Saved. File size: {size_mb:.1f} MB')

# Verify reload
loaded = np.load(FEATURES_NPZ)
print('    Verification (reloaded):')
for key in ['X_train','y_train','X_val','y_val','X_test','y_test']:
    arr = loaded[key]
    print(f'      {key:10s}  shape={str(arr.shape):22s}  dtype={arr.dtype}')
loaded.close()

# ── 9. Final summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('DONE - ResNet50 Feature Extraction Summary')
print('='*60)
print(f'  Model      : ResNet50 (ImageNet1K_V1 weights, head removed)')
print(f'  Feature dim: 2048')
print(f'  Device     : {device}')
print(f'  X_train    : {X_train.shape}   y_train: {y_train.shape}')
print(f'  X_val      : {X_val.shape}     y_val  : {y_val.shape}')
print(f'  X_test     : {X_test.shape}    y_test : {y_test.shape}')
print(f'  NPZ saved  : {FEATURES_NPZ}')
print(f'  Label map  : {LABEL_MAP_JSON}')
print('[OK] Feature extraction complete.')
