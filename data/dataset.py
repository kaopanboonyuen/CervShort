"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CervShort — Cervical Cytology Dataset                           ║
║                                                                              ║
║  Dataset: 25,412 digitised cytology patches from 5 independent laboratories ║
║  in Thailand, annotated per Bethesda 2014 guidelines.                       ║
║                                                                              ║
║  Categories: NILM · ASC-US · LSIL · HSIL · SCC                             ║
║                                                                              ║
║  Preprocessing:                                                              ║
║    • Macenko color normalization                                              ║
║    • 256×256 nuclei-centred patches                                          ║
║    • Patient-disjoint 70/15/15 train–val–test split                         ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
║  Ref    : CervShort Section 3.1                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple


# Bethesda 2014 class mapping
BETHESDA_CLASSES = {
    "NILM"  : 0,
    "ASC-US": 1,
    "LSIL"  : 2,
    "HSIL"  : 3,
    "SCC"   : 4,
}
CLASS_NAMES = list(BETHESDA_CLASSES.keys())

# Laboratory domains
LAB_DOMAINS = {"lab_A": 0, "lab_B": 1, "lab_C": 2, "lab_D": 3, "lab_E": 4}


class CervicalCytologyDataset(Dataset):
    """
    Multi-center cervical cytology dataset.

    Directory structure:
        root/
          lab_A/NILM/*.png
          lab_A/LSIL/*.png
          ...
          lab_E/SCC/*.png

    Args:
        root       : Path to dataset root directory
        split      : 'train', 'val', or 'test'
        transform  : Image transforms (if None, uses default)
        target_lab : If set, only load samples from this lab (OOD eval)
        split_file : Optional JSON with patient-disjoint splits
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_lab: Optional[str] = None,
        split_file: Optional[str] = None,
    ):
        super().__init__()
        self.root       = root
        self.split      = split
        self.transform  = transform or self._default_transform(split)

        self.samples: List[Tuple[str, int, int]] = []  # (path, label, domain)

        # Load or build split index
        if split_file and os.path.exists(split_file):
            self._load_from_split_file(split_file, split, target_lab)
        else:
            self._scan_directory(target_lab)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_transform(split: str):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    # ──────────────────────────────────────────────────────────────────────────

    def _scan_directory(self, target_lab: Optional[str] = None):
        """Scan directory structure to build sample list."""
        for lab_name, domain_id in LAB_DOMAINS.items():
            if target_lab and lab_name != target_lab:
                continue
            lab_dir = os.path.join(self.root, lab_name)
            if not os.path.isdir(lab_dir):
                continue
            for class_name, class_id in BETHESDA_CLASSES.items():
                class_dir = os.path.join(lab_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.samples.append((
                            os.path.join(class_dir, fname),
                            class_id,
                            domain_id,
                        ))

    def _load_from_split_file(
        self,
        split_file: str,
        split: str,
        target_lab: Optional[str],
    ):
        """Load patient-disjoint split from JSON index file."""
        with open(split_file) as f:
            index = json.load(f)
        for entry in index.get(split, []):
            lab = entry["lab"]
            if target_lab and lab != target_lab:
                continue
            self.samples.append((
                os.path.join(self.root, entry["path"]),
                BETHESDA_CLASSES[entry["label"]],
                LAB_DOMAINS[lab],
            ))

    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {
            "image"    : img,
            "label"    : torch.tensor(label,  dtype=torch.long),
            "domain_id": torch.tensor(domain, dtype=torch.long),
            "path"     : path,
        }


# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    split_file: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test dataloaders.

    Returns:
        dict with 'train', 'val', 'test' keys.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CervicalCytologyDataset(
            root=root, split=split, split_file=split_file
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


def build_per_lab_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "test",
) -> Dict[str, DataLoader]:
    """
    Build one dataloader per lab for cross-center evaluation (Table 1).

    Returns:
        dict mapping lab name → DataLoader
    """
    per_lab = {}
    for lab in LAB_DOMAINS:
        ds = CervicalCytologyDataset(
            root=root, split=split, target_lab=lab
        )
        per_lab[lab] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return per_lab
