import os, sys, json, importlib.util, multiprocessing as mp, gc
from pathlib import Path
from datetime import datetime
from typing import List

PROJ_DIR = Path('taco_advanced_pipeline')
OUT_DIR = PROJ_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEPS_FLAG = OUT_DIR / ".deps_ok"

_PIP_REQUIREMENTS: List[str] = [
    "ultralytics",
    "timm",
    "albumentations",
    "opencv-python",
    "matplotlib",
    "pillow",
    "tqdm",
    "pandas",
    "scikit-learn",
    "transformers",
    "accelerate",
    "seaborn",
    "psutil",
    "pynvml",
    "tabulate",
    "requests",
    "torchvision",
]

_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}

_CHECK_IMPORTS = [
    "ultralytics", "timm", "albumentations", "cv2", "matplotlib",
    "PIL", "tqdm", "pandas", "sklearn", "transformers", "accelerate",
    "seaborn", "psutil", "pynvml", "tabulate","requests",
    "torchvision",
]

def _missing_imports() -> List[str]:
    missing = []
    for mod in _CHECK_IMPORTS:
        if importlib.util.find_spec(mod) is None:
            pip_name = _IMPORT_TO_PIP.get(mod, mod)
            if pip_name not in missing:
                missing.append(pip_name)
    return missing

def _install_packages(packages: List[str]) -> None:
    if not packages:
        return
    print(f"[DEPS] Instalando pacotes ausentes via pip: {packages}")
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("[DEPS] Falha ao instalar dependências.")
        raise e

def ensure_dependencies_once():
    """
    Garante dependências sem ficar preso ao .deps_ok antigo.
    - Ainda usa env var PIPELINE_DEPS_DONE (evita reentrância em workers).
    - Reinstala se FALTAR algo, mesmo com sentinela existente.
    - Escreve um hash simples da lista de reqs no sentinela.
    """
    import hashlib

    deps_fingerprint = hashlib.sha256(
        ("\n".join(sorted(_PIP_REQUIREMENTS)) + "\n" + sys.version).encode()
    ).hexdigest()

    if os.environ.get("PIPELINE_DEPS_DONE") == "1":
        missing = _missing_imports()
        if missing:
            _install_packages(missing)
        return

    sent_ok = False
    if DEPS_FLAG.exists():
        try:
            meta = json.load(open(DEPS_FLAG, "r", encoding="utf-8"))
            sent_ok = (meta.get("deps_fingerprint") == deps_fingerprint)
        except Exception:
            sent_ok = False

    missing = _missing_imports()
    if missing:
        _install_packages(missing)
        sent_ok = False

    if not sent_ok and mp.current_process().name == "MainProcess":
        DEPS_FLAG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEPS_FLAG, "w", encoding="utf-8") as f:
            json.dump({
                "done_at": datetime.now().isoformat(timespec="seconds"),
                "python": sys.version,
                "pip_requirements": _PIP_REQUIREMENTS,
                "deps_fingerprint": deps_fingerprint
            }, f, ensure_ascii=False, indent=2)

    os.environ["PIPELINE_DEPS_DONE"] = "1"

ensure_dependencies_once()

import os

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message='.*pynvml.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*albumentations.*')
warnings.filterwarnings('ignore')

import json as json_lib
import random
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_recall_fscore_support
)
from sklearn.utils import resample

from tqdm import tqdm

import os, time, psutil
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def get_system_usage():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu_info = None
    if torch.cuda.is_available():
        try:
            try:
                import nvidia_ml_py3 as pynvml
            except Exception:
                import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            except Exception:
                util = None
            gpu_info = {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "used_mb": float(mem.used) / (1024 ** 2),
                "total_mb": float(mem.total) / (1024 ** 2),
                "free_mb": float(mem.free) / (1024 ** 2),
                "util_percent": float(getattr(util, 'gpu', 0.0)) if util is not None else None
            }
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        except Exception:
            try:
                name = torch.cuda.get_device_name(0)
                total = None
                used = None
                try:
                    prop = torch.cuda.get_device_properties(0)
                    total = float(prop.total_memory) / (1024 ** 2)
                except Exception:
                    total = None
                try:
                    used = float(torch.cuda.memory_allocated(0)) / (1024 ** 2)
                except Exception:
                    used = None
                gpu_info = {
                    "name": name,
                    "used_mb": used,
                    "total_mb": total,
                    "free_mb": (total - used) if (total is not None and used is not None) else None,
                    "util_percent": None
                }
            except Exception:
                gpu_info = None
    return {"cpu_percent": cpu, "ram_percent": ram, "gpu": gpu_info}

def download_and_extract_taco(data_dir: Path, zip_path: Path, url: str) -> Path:
    """Baixa e extrai o dataset TACO."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"[INFO] Baixando TACO...")
        import requests
        response = requests.get(url)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    else:
        print("[INFO] Zip já existe.")
    extract_dir = data_dir / 'TACO'
    if not extract_dir.exists():
        print("[INFO] Extraindo...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    else:
        print("[INFO] Já extraído.")
    return extract_dir

def load_coco_annotations(taco_dir: Path) -> dict:
    """Carrega annotations.json."""
    ann_path = None
    for p in taco_dir.rglob('annotations.json'):
        ann_path = p
        break
    assert ann_path is not None, "annotations.json não encontrado"
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json_lib.load(f)
    return coco

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def create_crops_with_segmentation(coco: dict, taco_dir: Path,
                                   output_dir: Path, margin: float = 0.15):
    """Gera crops com margem maior para contexto."""
    images = {img['id']: img for img in coco['images']}
    categories = {c['id']: c for c in coco['categories']}
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for ann in tqdm(coco['annotations'], desc='Gerando crops'):
        img_id = ann['image_id']
        img_info = images[img_id]
        img_rel_path = img_info['file_name']
        img_path = next(iter(taco_dir.glob(f"**/{img_rel_path}")), None)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        segs = ann.get('segmentation', [])

        if not segs:
            x, y, bw, bh = ann['bbox']
            x1 = clamp(int(x), 0, w-1)
            y1 = clamp(int(y), 0, h-1)
            x2 = clamp(int(x + bw), 1, w)
            y2 = clamp(int(y + bh), 1, h)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in segs:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], color=255)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        dx = int((x_max - x_min) * margin)
        dy = int((y_max - y_min) * margin)

        x1 = clamp(x_min - dx, 0, w-1)
        y1 = clamp(y_min - dy, 0, h-1)
        x2 = clamp(x_max + dx, 1, w)
        y2 = clamp(y_max + dy, 1, h)

        crop = img[y1:y2, x1:x2]
        cat_id = ann['category_id']
        cat_name = categories[cat_id]['name']
        class_dir = output_dir / cat_name
        class_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f"{Path(img_rel_path).stem}_{ann['id']}.jpg"
        crop_path = class_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)

        records.append({
            'path': str(crop_path),
            'class_name': cat_name,
            'class_id': cat_id
        })

    df = pd.DataFrame(records)
    class_map = {cid: i for i, cid in enumerate(sorted(df['class_id'].unique()))}
    df['class_id'] = df['class_id'].map(class_map)
    return df

def get_advanced_transforms(img_size, MEAN, STD):
    """Augmentations state-of-the-art com Albumentations."""
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),

        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ], p=0.8),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.2),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
        ], p=0.3),

        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5),
        ], p=0.2),

        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.3),

        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return train_transform, val_test_transform

def mixup_data(x, y, alpha=1.0):
    """Aplica MixUp."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Gera bbox aleatório para CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """Aplica CutMix."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

class AdvancedTACODataset(Dataset):
    """Dataset com Albumentations."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['class_id']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

class PrefetchLoader:
    """
    Wrapper around a DataLoader that:
    - pré-carrega o próximo batch para a GPU em um stream separado (quando em CUDA)
    - delega atributos comuns para o loader interno (len, dataset, batch_size, etc.)
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if (self.device.type == 'cuda') else None
        self.iter = None
        self.next_inputs = None
        self.next_labels = None

    def __len__(self):
        try:
            return len(self.loader)
        except Exception:
            return 0

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream:
            self.preload()
        return self

    def preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_inputs = None
            self.next_labels = None
            return
        inputs, labels = batch
        if self.stream:
            with torch.cuda.stream(self.stream):
                self.next_inputs = inputs.to(self.device, non_blocking=True)
                self.next_labels = labels.to(self.device, non_blocking=True)
        else:
            self.next_inputs = inputs
            self.next_labels = labels

    def __next__(self):
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            inputs = self.next_inputs
            labels = self.next_labels
            if inputs is None:
                raise StopIteration
            self.preload()
            return inputs, labels
        else:
            return next(self.iter)

    def __getattr__(self, name):
        try:
            return getattr(self.loader, name)
        except AttributeError:
            raise

class AdvancedClassifier(nn.Module):
    """Wrapper para modelos com dropout e regularização."""
    def __init__(self, model_name, num_classes, dropout=0.3, pretrained=True):
        super().__init__()
        self.model_name = model_name

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features

        if 'vit' in model_name:
            self.head = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Dropout(dropout),
                nn.Linear(n_features, n_features // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(n_features // 2, num_classes)
            )
        elif 'swin' in model_name or 'convnext' in model_name:
            self.head = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Dropout(dropout),
                nn.Linear(n_features, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(n_features, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def create_advanced_model(model_name, num_classes):
    """Factory para criar modelos avançados."""
    print(f"\n[INFO] Criando modelo: {model_name}")
    model = AdvancedClassifier(model_name, num_classes, dropout=0.3, pretrained=True)
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model, unfreeze_layers=-1):
    """Descongela camadas para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print(f"[INFO] Modelo totalmente descongelado para fine-tuning")

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy com Label Smoothing."""
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction='mean')
        return (1 - self.epsilon) * nll + self.epsilon * (loss / n_classes)

def train_epoch_advanced(model, loader, criterion, optimizer, device,
                         use_mixup=True, use_cutmix=True, MIXUP_ALPHA=0.2, CUTMIX_ALPHA=1.0,
                         scaler=None):
    """Treina uma época com MixUp/CutMix. Suporta AMP via scaler (se fornecido)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    use_amp = (scaler is not None)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        r = np.random.rand()
        if use_mixup and r < 0.3:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            mixed = True
        elif use_cutmix and r < 0.6:
            inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            if mixed:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)

        if not mixed:
            correct += predicted.eq(labels).sum().item()
        else:
            correct += (lam * predicted.eq(labels_a).sum().item() +
                        (1 - lam) * predicted.eq(labels_b).sum().item())

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'acc': f'{100.*correct/total:.2f}%'})

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100. * correct / max(1, total)
    return epoch_loss, epoch_acc

def validate_epoch_advanced(model, loader, criterion, device):
    """Valida o modelo."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = 100. * accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def train_advanced_model(model, model_name, train_loader, val_loader,
                         EPOCHS, WARMUP_EPOCHS, BASE_LR, WEIGHT_DECAY,
                         MIN_LR, LABEL_SMOOTHING, PATIENCE, device,
                         MIXUP_ALPHA=0.2, CUTMIX_ALPHA=1.0):
    """Pipeline de treinamento em 2 fases com monitoramento de tempo/recursos e histórico detalhado."""
    import time
    from pathlib import Path

    def _safe_import(name):
        try:
            return __import__(name)
        except Exception:
            return None

    def _system_usage_snapshot():
        try:
            return get_system_usage()
        except Exception:
            return {"cpu_percent": None, "ram_percent": None, "gpu": None}

    print(f"\n{'='*70}")
    print(f"TREINANDO: {model_name.upper()}")
    print(f"{'='*70}\n")

    criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    MODELS_DIR = Path('taco_advanced_pipeline') / 'outputs' / 'models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'epoch_time_sec': [],
        'lr': [],
        'phase': [],
        'train_start_usage': None,
        'train_end_usage': None,
        'train_total_time_sec': None,
    }

    print("[FASE 1] Treinando classificador (backbone congelado)...")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    num_steps = len(train_loader) * max(1, (EPOCHS // 3))
    warmup_steps = len(train_loader) * max(0, WARMUP_EPOCHS)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=BASE_LR, total_steps=max(1, num_steps),
        pct_start=min(0.9, warmup_steps / max(1, num_steps)), anneal_strategy='cos'
    )

    best_f1 = 0.0
    patience_counter = 0

    phase1_epochs = max(1, EPOCHS // 3)

    train_start_time = time.time()
    history['train_start_usage'] = _system_usage_snapshot()

    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch + 1}/{phase1_epochs} (fase 1)")
        epoch_t0 = time.time()

        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(epoch >= WARMUP_EPOCHS),
            use_cutmix=(epoch >= WARMUP_EPOCHS),
            MIXUP_ALPHA=MIXUP_ALPHA, CUTMIX_ALPHA=CUTMIX_ALPHA,
            scaler=scaler
        )

        val_loss, val_acc, val_f1, _, _ = validate_epoch_advanced(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['epoch_time_sec'].append(time.time() - epoch_t0)
        try:
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            history['lr'].append(sum(current_lrs)/len(current_lrs))
        except Exception:
            history['lr'].append(None)
        history['phase'].append('phase1')

        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_phase1_best.pth')
            print(f"✓ Melhor modelo (fase 1) salvo — F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping (fase 1)")
            break

        try:
            scheduler.step()
        except Exception:
            pass

    model.load_state_dict(torch.load(MODELS_DIR / f'{model_name}_phase1_best.pth', map_location=device))

    print(f"\n[FASE 2] Fine-tuning completo...")
    unfreeze_model(model)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR / 10, weight_decay=WEIGHT_DECAY)

    phase2_epochs = max(1, (EPOCHS * 2) // 3)
    num_steps = len(train_loader) * phase2_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_steps), eta_min=MIN_LR)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(phase2_epochs):
        print(f"\nEpoch {epoch + 1}/{phase2_epochs} (fase 2)")
        epoch_t0 = time.time()

        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device,
            use_mixup=True, use_cutmix=True,
            MIXUP_ALPHA=MIXUP_ALPHA, CUTMIX_ALPHA=CUTMIX_ALPHA,
            scaler=scaler
        )

        val_loss, val_acc, val_f1, _, _ = validate_epoch_advanced(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['epoch_time_sec'].append(time.time() - epoch_t0)
        try:
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            history['lr'].append(sum(current_lrs)/len(current_lrs))
        except Exception:
            history['lr'].append(None)
        history['phase'].append('phase2')

        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_final_best.pth')
            print(f"✓ Melhor modelo (fase 2) salvo — F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping (fase 2)")
            break

        try:
            scheduler.step()
        except Exception:
            pass

    model.load_state_dict(torch.load(MODELS_DIR / f'{model_name}_final_best.pth', map_location=device))

    history['train_end_usage'] = _system_usage_snapshot()
    history['train_total_time_sec'] = time.time() - train_start_time

    tt_min = (history['train_total_time_sec'] or 0) / 60
    su = history['train_start_usage'] or {}
    eu = history['train_end_usage'] or {}
    print("\n📊 [Resumo de Desempenho]")
    print(f"⏱️  Tempo total de treino: {tt_min:.2f} min")
    print(f"🧠  CPU: {su.get('cpu_percent')}% → {eu.get('cpu_percent')}%")
    print(f"💾  RAM: {su.get('ram_percent')}% → {eu.get('ram_percent')}%")
    gpu = eu.get('gpu') if isinstance(eu, dict) else None
    if gpu:
        try:
            name = gpu.get('name', 'GPU')
            used = gpu.get('used_mb')
            total = gpu.get('total_mb')
            util = gpu.get('util_percent')
            used_s = f"{used:.1f} MB" if used is not None else "N/A"
            total_s = f"{total:.1f} MB" if total is not None else "N/A"
            util_s = f"{util:.1f}%" if util is not None else "N/A"
            print(f"🎮  GPU: {name} | Mem: {used_s} / {total_s} | Util: {util_s}")
        except Exception:
            pass

    return model, history

def evaluate_advanced(model, test_loader, class_names, model_name, device, PLOTS_DIR: Path):
    """Avaliação completa."""
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO - {model_name.upper()}")
    print(f"{'='*70}\n")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}\n")

    print("Relatório por Classe:")
    print(classification_report(all_labels, all_preds,
                               target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f'{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

    pred_vs_true_path = None
    try:
        image_paths = None
        if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'df'):
            image_paths = test_loader.dataset.df['path'].tolist()

        if image_paths is not None and len(image_paths) == len(all_preds):
            pred_vs_true_path = visualize_predictions_grid(
                image_paths, all_preds, all_labels, all_probs,
                class_names, model_name, PLOTS_DIR, n_images=25, incorrect_only=False
            )
    except Exception:
        pred_vs_true_path = None

    roc_path = None
    pr_path = None
    calibration_path = None
    per_class_path = None

    try:
        roc_path = plot_roc_curves(all_labels, all_probs, class_names, model_name, PLOTS_DIR)
    except Exception as e:
        print(f"Erro ao gerar ROC curves: {e}")

    try:
        pr_path = plot_precision_recall_curves(all_labels, all_probs, class_names, model_name, PLOTS_DIR)
    except Exception as e:
        print(f"Erro ao gerar Precision-Recall curves: {e}")

    try:
        calibration_path = plot_calibration_curve(all_labels, all_probs, model_name, PLOTS_DIR)
    except Exception as e:
        print(f"Erro ao gerar Calibration curve: {e}")

    try:
        per_class_path = plot_per_class_metrics(all_labels, all_preds, class_names, model_name, PLOTS_DIR)
    except Exception as e:
        print(f"Erro ao gerar Per-class metrics: {e}")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'pred_vs_true_path': str(pred_vs_true_path) if pred_vs_true_path is not None else None,
        'roc_curves_path': str(roc_path) if roc_path is not None else None,
        'pr_curves_path': str(pr_path) if pr_path is not None else None,
        'calibration_path': str(calibration_path) if calibration_path is not None else None,
        'per_class_metrics_path': str(per_class_path) if per_class_path is not None else None
    }

def visualize_predictions_grid(image_paths, preds, labels, probs, class_names,
                               model_name, PLOTS_DIR: Path, n_images=25, incorrect_only=False):
    """Salva um grid com imagens anotadas pelos rótulos predito/real e probabilidades (versão concisa)."""
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    import math, cv2, numpy as np
    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs) if probs is not None else None

    indices = np.arange(len(preds))
    if incorrect_only:
        indices = indices[preds != labels]

    if len(indices) == 0:
        return None

    indices = indices[:n_images]
    k = len(indices)
    cols = min(5, k)
    rows = math.ceil(k / cols)

    sns.set_style('white')
    sns.set_palette('colorblind')
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array(axes)[:, None]

    axes = axes.reshape(-1)

    for ax_i, idx in enumerate(indices):
        ax = axes[ax_i]
        img_path = None
        try:
            img_path = image_paths[idx]
        except Exception:
            img_path = None

        if img_path is None:
            ax.axis('off')
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            ax.axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = int(preds[idx])
        lab = int(labels[idx])
        prob = None
        try:
            prob = float(probs[idx][pred]) if (probs is not None and probs.ndim == 2) else (float(probs[idx]) if probs is not None else None)
        except Exception:
            prob = None

        title_color = 'green' if pred == lab else 'red'
        short_gt = class_names[lab] if lab < len(class_names) else str(lab)
        short_pred = class_names[pred] if pred < len(class_names) else str(pred)
        title = f"GT: {short_gt}\nPRED: {short_pred}"
        if prob is not None:
            title += f" ({prob:.2f})"

        ax.imshow(img)
        ax.set_title(title, color=title_color, fontsize=9)
        ax.axis('off')

    for j in range(k, len(axes)):
        try:
            axes[j].axis('off')
        except Exception:
            pass

    plt.suptitle(f'{model_name} — Predictions (n={k})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = PLOTS_DIR / f'{model_name}_pred_vs_true.png'
    try:
        plt.savefig(out_path, dpi=200)
    except Exception:
        pass
    plt.close()
    return out_path

def plot_training_history(history, model_name, PLOTS_DIR: Path):
    """Plota curvas de treinamento de forma concisa e legível."""
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get('train_loss', [])) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss', marker='o')
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get('train_acc', []), label='Train Acc', marker='o')
    axes[1].plot(epochs, history.get('val_acc', []), label='Val Acc', marker='o')
    if history.get('val_f1'):
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history.get('val_f1', []), label='Val F1', color='tab:green', marker='s')
        ax2.set_ylabel('Val F1', color='tab:green')
        for tl in ax2.get_yticklabels():
            tl.set_color('tab:green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} — Accuracy / F1')
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{model_name}_training_history.png', dpi=200)
    plt.close()

def plot_comparison(results, PLOTS_DIR: Path):
    """Gráfico conciso de comparação entre modelos (Accuracy e Macro F1)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style('whitegrid')
    palette = sns.color_palette('colorblind')

    models = list(results.keys())
    accuracies = [r['accuracy'] * 100 for r in results.values()]
    f1_scores = [r['f1_macro'] for r in results.values()]

    order = np.argsort(accuracies)[::-1]
    models_ord = [models[i] for i in order]
    acc_ord = [accuracies[i] for i in order]
    f1_ord = [f1_scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.6)))
    y_pos = np.arange(len(models_ord))
    bars = ax.barh(y_pos, acc_ord, color=palette[:len(models_ord)], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_ord)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Comparison — Accuracy (with Macro F1)')

    for i, (b, acc_val, f1_val) in enumerate(zip(bars, acc_ord, f1_ord)):
        w = b.get_width()
        ax.text(w + 0.5, b.get_y() + b.get_height() / 2, f'{w:.2f}%', va='center', fontsize=9)
        ax.text(95, b.get_y() + b.get_height() / 2 - 0.15 * b.get_height(), f'F1: {f1_val:.3f}', va='center', fontsize=9, color='gray')

    ax.set_xlim(0, max(100, max(acc_ord) + 5))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'models_comparison.png', dpi=200)
    plt.close()

def plot_roc_curves(labels, probs, class_names, model_name, PLOTS_DIR: Path):
    """ROC curves (micro / macro + top-3 class curves) com legenda enxuta."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        labels_arr = np.array(labels)
        probs_arr = np.array(probs)
        if probs_arr.ndim != 2:
            return None
        n_classes = len(class_names)
        y_bin = label_binarize(labels_arr, classes=list(range(n_classes)))

        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs_arr.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr["micro"], tpr["micro"], label=f'micro (AUC={roc_auc["micro"]:.3f})', color='black', lw=2)
        plt.plot(fpr["macro"], tpr["macro"], label=f'macro (AUC={roc_auc["macro"]:.3f})', color='navy', lw=2)

        topk = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)[:3]
        colors = sns.color_palette('tab10')
        for i, c in zip(topk, colors):
            plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})', lw=1.5)

        plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = PLOTS_DIR / f'{model_name}_roc_curves.png'
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None

def plot_precision_recall_curves(labels, probs, class_names, model_name, PLOTS_DIR: Path):
    """Precision-Recall concise: micro-average + top-3 classes."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        labels_arr = np.array(labels)
        probs_arr = np.array(probs)
        if probs_arr.ndim != 2:
            return None
        n_classes = len(class_names)
        y_bin = label_binarize(labels_arr, classes=list(range(n_classes)))

        precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), probs_arr.ravel())
        ap_micro = average_precision_score(y_bin, probs_arr, average='micro')

        plt.figure(figsize=(8, 6))
        plt.plot(recall_micro, precision_micro, label=f'micro (AP={ap_micro:.3f})', color='black', lw=2)

        aps = {}
        for i in range(n_classes):
            p, r, _ = precision_recall_curve(y_bin[:, i], probs_arr[:, i])
            aps[i] = average_precision_score(y_bin[:, i], probs_arr[:, i])
        topk = sorted(aps.keys(), key=lambda i: aps[i], reverse=True)[:3]
        colors = sns.color_palette('tab10')
        for i, c in zip(topk, colors):
            p, r, _ = precision_recall_curve(y_bin[:, i], probs_arr[:, i])
            plt.plot(r, p, label=f'{class_names[i]} (AP={aps[i]:.3f})', lw=1.5)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall - {model_name}')
        plt.legend(loc='lower left', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = PLOTS_DIR / f'{model_name}_pr_curves.png'
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None

def plot_calibration_curve(labels, probs, model_name, PLOTS_DIR: Path, n_bins=10):
    """Calibration plot (reliability diagram) com ECE anotado."""
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        labels_arr = np.array(labels)
        probs_arr = np.array(probs)
        if probs_arr.ndim != 2:
            return None

        confidences = probs_arr.max(axis=1)
        predictions = probs_arr.argmax(axis=1)
        accuracies = (predictions == labels_arr).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_conf = []; bin_acc = []; bin_counts = []
        for i in range(n_bins):
            mask = bin_indices == i
            cnt = mask.sum()
            if cnt > 0:
                bin_conf.append(confidences[mask].mean())
                bin_acc.append(accuracies[mask].mean())
                bin_counts.append(cnt)
            else:
                bin_conf.append((bins[i] + bins[i+1]) / 2.0)
                bin_acc.append(0.0)
                bin_counts.append(0)

        total = sum(bin_counts) if sum(bin_counts) > 0 else 1
        ece = sum([abs(a - c) * cnt for a, c, cnt in zip(bin_acc, bin_conf, bin_counts)]) / total

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.bar(bin_conf, bin_acc, width=1.0 / n_bins * 0.9, alpha=0.7, edgecolor='k')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration - {model_name} (ECE={ece:.4f})')
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        out_path = PLOTS_DIR / f'{model_name}_calibration.png'
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None

def plot_per_class_metrics(labels, preds, class_names, model_name, PLOTS_DIR: Path):
    """Plota métricas por classe (precision, recall, f1) com ordenação por suporte e cores legíveis."""
    from sklearn.metrics import precision_recall_fscore_support
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        order = np.argsort(support)[::-1]
        names_ord = [class_names[i] for i in order]
        prec_ord = precision[order]; rec_ord = recall[order]; f1_ord = f1[order]; sup_ord = support[order]

        x = np.arange(len(names_ord))
        width = 0.28
        fig, ax = plt.subplots(figsize=(max(10, len(names_ord)*0.25), 6))
        ax.bar(x - width, prec_ord, width, label='Precision', color='tab:blue', alpha=0.9)
        ax.bar(x, rec_ord, width, label='Recall', color='tab:orange', alpha=0.9)
        ax.bar(x + width, f1_ord, width, label='F1', color='tab:green', alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(names_ord, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Per-Class Metrics - {model_name}')
        ax.set_ylabel('Score')
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.25)

        for i, s in enumerate(sup_ord):
            ax.text(i, -0.05, f'n={int(s)}', ha='center', va='top', fontsize=8, color='gray')

        plt.tight_layout()
        out_path = PLOTS_DIR / f'{model_name}_per_class_metrics.png'
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path
    except Exception:
        return None

def ensemble_predict(models, loader, device, weights=None):
    """
    Faz ensemble por média (ponderada) das probabilidades softmax.
    Retorna (preds, labels, probs) — listas numpy.
    """
    import math
    for m in models:
        m.eval()
        try:
            m.to(device)
        except Exception:
            pass

    if weights is None:
        weights = [1.0] * len(models)
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Ensemble Predict'):
            try:
                inputs, labels = batch
            except Exception:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                else:
                    raise

            inputs = inputs.to(device, non_blocking=True)
            probs_sum = None
            for w, m in zip(weights, models):
                out = m(inputs)
                p = torch.softmax(out, dim=1)
                if probs_sum is None:
                    probs_sum = w * p
                else:
                    probs_sum = probs_sum + w * p
            preds = probs_sum.argmax(dim=1)

            try:
                labels_cpu = labels.cpu().numpy()
            except Exception:
                labels_cpu = np.asarray(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_cpu)
            all_probs.extend(probs_sum.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    import time, json, platform, shutil
    from datetime import datetime

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    IMG_SIZE = 384
    BATCH_SIZE = 16
    cpu_count = os.cpu_count() or 8
    NUM_WORKERS = int(os.environ.get("TACO_NUM_WORKERS", min(8, max(2, cpu_count - 1))))
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    EPOCHS = 60
    WARMUP_EPOCHS = 5
    BASE_LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    PATIENCE = 15

    PROJ_DIR = Path('taco_advanced_pipeline')
    DATA_DIR = PROJ_DIR / 'datasets'
    OUT_DIR = PROJ_DIR / 'outputs'
    TACO_ZIP = DATA_DIR / 'TACO.zip'
    TACO_URL = 'https://zenodo.org/records/3354286/files/TACO.zip?download=1'
    CROPS_DIR = PROJ_DIR / 'taco_crops'
    MODELS_DIR = OUT_DIR / 'models'
    PLOTS_DIR = OUT_DIR / 'plots'
    for d in [DATA_DIR, OUT_DIR, MODELS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        import psutil
    except Exception:
        psutil = None
    try:
        try:
            import nvidia_ml_py3 as pynvml
        except ImportError:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                import pynvml
    except ImportError:
        pynvml = None

    def sys_snapshot():
        snap = {
            "cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
            "ram_percent": psutil.virtual_memory().percent if psutil else None,
            "disk_free_gb": round(shutil.disk_usage(str(PROJ_DIR)).free / 1e9, 2) if shutil else None,
            "gpu_name": None,
            "gpu_mem_used_mb": None
        }
        if torch.cuda.is_available():
            try:
                if pynvml:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snap["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
                    snap["gpu_mem_used_mb"] = int(mem.used / (1024**2))
                    pynvml.nvmlShutdown()
                else:
                    snap["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return snap

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
        print(f"[INFO] CuDNN: {torch.backends.cudnn.version()}")

    env_info = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": getattr(torchvision, '__version__', None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    print("\n[AMBIENTE]")
    print(json.dumps(env_info, indent=2, ensure_ascii=False))

    print("\n" + "="*70)
    print("PIPELINE AVANÇADO - VISION TRANSFORMERS + TÉCNICAS MODERNAS")
    print("="*70 + "\n")

    t0_total = time.perf_counter()
    run_start_snap = sys_snapshot()

    t0 = time.perf_counter()
    print("[ETAPA 1] Preparando dados...")
    crops_df_path = CROPS_DIR / 'crops_dataframe.csv'

    if CROPS_DIR.exists() and any(CROPS_DIR.iterdir()) and crops_df_path.exists():
        print("[INFO] Carregando crops existentes...")
        df_crops = pd.read_csv(crops_df_path)
    else:
        print("[INFO] Gerando crops...")
        taco_dir = download_and_extract_taco(DATA_DIR, TACO_ZIP, TACO_URL)
        coco = load_coco_annotations(taco_dir)
        df_crops = create_crops_with_segmentation(coco, taco_dir, CROPS_DIR, margin=0.15)
        df_crops.to_csv(crops_df_path, index=False)
    prep_time = time.perf_counter() - t0

    print(f"\nTotal de crops: {len(df_crops)}")
    print(f"Número de classes: {df_crops['class_id'].nunique()}")

    t0 = time.perf_counter()
    print("\n[ETAPA 2] Balanceamento de classes...")
    class_counts = df_crops['class_id'].value_counts()
    MIN_SAMPLES = 6
    print(f"Classes com < {MIN_SAMPLES} amostras: {(class_counts < MIN_SAMPLES).sum()}")

    dfs_to_concat = []
    for class_id in df_crops['class_id'].unique():
        class_df = df_crops[df_crops['class_id'] == class_id]
        n_samples = len(class_df)

        if n_samples < MIN_SAMPLES:
            oversampled = resample(class_df, n_samples=MIN_SAMPLES,
                                   replace=True, random_state=SEED)
            dfs_to_concat.append(oversampled)
            print(f"  ✓ Classe {class_id}: {n_samples} → {MIN_SAMPLES}")
        else:
            dfs_to_concat.append(class_df)

    df_crops = pd.concat(dfs_to_concat, ignore_index=True)
    class_map = {old_id: new_id for new_id, old_id in enumerate(sorted(df_crops['class_id'].unique()))}
    df_crops['class_id'] = df_crops['class_id'].map(class_map)
    balance_time = time.perf_counter() - t0

    print(f"Total após balanceamento: {len(df_crops)}")

    t0 = time.perf_counter()
    print("\n[ETAPA 3] Dividindo dados (70/15/15)...")
    try:
        train_df, temp_df = train_test_split(df_crops, test_size=0.3,
                                             stratify=df_crops['class_id'],
                                             random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                           stratify=temp_df['class_id'],
                                           random_state=SEED)
    except ValueError:
        print("Usando split não estratificado...")
        train_df, temp_df = train_test_split(df_crops, test_size=0.3, random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
    split_time = time.perf_counter() - t0

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    t0 = time.perf_counter()
    print("\n[ETAPA 4] Criando datasets com augmentations avançados...")
    train_transform, val_test_transform = get_advanced_transforms(IMG_SIZE, MEAN, STD)

    train_dataset = AdvancedTACODataset(train_df, transform=train_transform)
    val_dataset = AdvancedTACODataset(val_df, transform=val_test_transform)
    test_dataset = AdvancedTACODataset(test_df, transform=val_test_transform)

    class_counts = Counter(train_df['class_id'])
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_df['class_id']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else 1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else 1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else 1
    )
    if device.type == 'cuda':
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)
        test_loader = PrefetchLoader(test_loader, device)
    loaders_time = time.perf_counter() - t0

    num_classes = df_crops['class_id'].nunique()
    class_names = sorted(df_crops['class_name'].unique())

    print(f"Número de classes: {num_classes}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Num workers: {NUM_WORKERS} (persistent_workers={'True' if NUM_WORKERS > 0 else 'False'})")

    xb, yb = next(iter(train_loader))
    print(f"[SANITY] batch imgs: {tuple(xb.shape)}, dtype={xb.dtype}, device={xb.device}, labels shape={tuple(yb.shape)}")

    del xb, yb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[ETAPA 5] Treinando modelos state-of-the-art...")

    models_config = [
        'vit_base_patch16_384',
        'swin_base_patch4_window12_384',
        'convnext_base',
    ]

    results = {}
    trained_models = []
    per_model_times = {}

    for model_name in models_config:
        print(f"\n{'#' * 70}")
        print(f"# MODELO: {model_name.upper()}")
        print(f"{'#' * 70}\n")

        try:
            model = create_advanced_model(model_name, num_classes).to(device)

            t_model0 = time.perf_counter()
            model, history = train_advanced_model(
                model, model_name, train_loader, val_loader,
                EPOCHS, WARMUP_EPOCHS, BASE_LR, WEIGHT_DECAY,
                MIN_LR, LABEL_SMOOTHING, PATIENCE, device,
                MIXUP_ALPHA=MIXUP_ALPHA, CUTMIX_ALPHA=CUTMIX_ALPHA
            )
            per_model_times[model_name] = time.perf_counter() - t_model0

            try:
                hist_dict = {k: v for k, v in history.items() if isinstance(v, (list, tuple, np.ndarray))}
                if hist_dict:
                    hist_df = pd.DataFrame(hist_dict)
                    hist_df.to_csv(OUT_DIR / f'{model_name}_history.csv', index=False)
                else:
                    with open(OUT_DIR / f'{model_name}_history.json', 'w', encoding='utf-8') as hf:
                        json.dump(history, hf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Não consegui salvar history CSV de {model_name}: {e}")

            plot_training_history(history, model_name, PLOTS_DIR)

            eval_results = evaluate_advanced(
                model, test_loader, class_names, model_name, device, PLOTS_DIR
            )
            if history.get('epoch_time_sec'):
                avg_epoch_s = float(np.mean(history['epoch_time_sec']))
                imgs_per_epoch = len(train_dataset)
                eval_results['approx_throughput_imgs_per_sec'] = imgs_per_epoch / max(1e-9, avg_epoch_s)
            results[model_name] = eval_results
            trained_models.append(model)

            print(f"\n✓ {model_name} COMPLETO!  (tempo: {per_model_times[model_name]:.1f}s)")

            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            time.sleep(1)

        except Exception as e:
            print(f"\n⚠️  Erro ao treinar {model_name}: {e}")
            print("Continuando com próximo modelo...")
            continue

    if len(trained_models) > 1:
        print("\n[ETAPA 6] Criando ensemble de modelos...")

        weights = [results[name]['f1_macro'] for name in results.keys()]

        ensemble_preds, ensemble_labels, _ = ensemble_predict(
            trained_models, test_loader, device, weights=weights
        )

        ensemble_acc = accuracy_score(ensemble_labels, ensemble_preds)
        ensemble_f1_macro = f1_score(ensemble_labels, ensemble_preds, average='macro')
        ensemble_f1_weighted = f1_score(ensemble_labels, ensemble_preds, average='weighted')

        results['ensemble'] = {
            'accuracy': ensemble_acc,
            'f1_macro': ensemble_f1_macro,
            'f1_weighted': ensemble_f1_weighted,
            'predictions': ensemble_preds,
            'labels': ensemble_labels
        }

        print(f"\n🎯 ENSEMBLE RESULTS:")
        print(f"   Accuracy: {ensemble_acc * 100:.2f}%")
        print(f"   Macro F1: {ensemble_f1_macro:.4f}")
        print(f"   Weighted F1: {ensemble_f1_weighted:.4f}")

    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL DE MODELOS")
    print("=" * 70 + "\n")

    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy (%)': res['accuracy'] * 100,
            'Macro F1': res['f1_macro'],
            'Weighted F1': res['f1_weighted'],
            'Throughput (img/s)': res.get('approx_throughput_imgs_per_sec', None),
            'Train Time (s)': per_model_times.get(name, None)
        })

    comparison_df = pd.DataFrame(comparison_data)
    if comparison_df.empty:
        print("[WARN] Nenhum resultado de modelo foi gerado. Verifique erros anteriores.")
    else:
        if 'Macro F1' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Macro F1', ascending=False)
        elif 'Accuracy (%)' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Accuracy (%)', ascending=False)

    print(comparison_df.to_string(index=False) if not comparison_df.empty else "No model results to show.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUT_DIR / 'models_comparison_advanced.csv', index=False)

    if len(results) > 1 and 'ensemble' in results:
        plot_comparison({k: v for k, v in results.items() if k != 'ensemble'}, PLOTS_DIR)

    if not comparison_df.empty:
        best_model_name = comparison_df.iloc[0]['Model']
        best_f1 = comparison_df.iloc[0].get('Macro F1', None)
    else:
        best_model_name = None
        best_f1 = None

    total_time = time.perf_counter() - t0_total
    run_end_snap = sys_snapshot()

    if best_model_name:
        print(f"\n🏆 MELHOR MODELO: {best_model_name.upper()}")
        if best_f1 is not None:
            print(f"     Macro F1: {best_f1:.4f}")
        print(f"   Accuracy: {comparison_df.iloc[0]['Accuracy (%)']:.2f}%")
    else:
        print("\n🏆 Nenhum modelo treinado com sucesso.")

    print("\n[RESUMO DE EXECUÇÃO]")
    print(f"Prep dados:      {prep_time:.1f}s")
    print(f"Balanceamento:   {balance_time:.1f}s")
    print(f"Split:           {split_time:.1f}s")
    print(f"Loaders:         {loaders_time:.1f}s")
    for m, secs in per_model_times.items():
        print(f"Treino {m:>28}: {secs:.1f}s")
    print(f"Tempo total:     {total_time:.1f}s")

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETO!")
    print(f"{'=' * 70}")
    print(f"✓ Modelos salvos em: {MODELS_DIR}")
    print(f"✓ Gráficos salvos em: {PLOTS_DIR}")
    print(f"✓ Resultados salvos em: {OUT_DIR}")

    run_cfg = {
        "env": env_info,
        "hyperparams": {
            "IMG_SIZE": IMG_SIZE, "BATCH_SIZE": BATCH_SIZE, "NUM_WORKERS": NUM_WORKERS,
            "EPOCHS": EPOCHS, "WARMUP_EPOCHS": WARMUP_EPOCHS, "BASE_LR": BASE_LR,
            "MIN_LR": MIN_LR, "WEIGHT_DECAY": WEIGHT_DECAY, "LABEL_SMOOTHING": LABEL_SMOOTHING,
            "MIXUP_ALPHA": MIXUP_ALPHA, "CUTMIX_ALPHA": CUTMIX_ALPHA, "PATIENCE": PATIENCE,
            "MEAN": MEAN, "STD": STD
        },
        "timings_sec": {
            "prep": prep_time,
            "balance": balance_time,
            "split": split_time,
            "loaders": loaders_time,
            "per_model": per_model_times,
            "total": total_time
        },
        "dataset": {
            "num_classes": int(num_classes),
            "total_crops": int(len(df_crops)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df))
        },
        "system_usage_start": run_start_snap,
        "system_usage_end": run_end_snap,
        "results": results,
        "comparison": comparison_df.to_dict(orient='records') if not comparison_df.empty else []
    }

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)
        print(f"[INFO] run_config salvo em: {OUT_DIR / 'run_config.json'}")
    except Exception as e:
        print(f"[WARN] Não consegui salvar run_config.json: {e}")

    return results, comparison_df

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    results, comparison = main()
