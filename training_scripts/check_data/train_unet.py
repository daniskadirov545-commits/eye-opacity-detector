import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================= CONFIG =================
@dataclass
class CFG:
    images_dir: str = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
    masks_dir: str  = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks"

    out_dir: str = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\models"
    model_name: str = "unet_corneal_opacity.pt"

    img_size: int = 256
    batch_size: int = 8
    epochs: int = 25
    lr: float = 1e-3

    val_split: float = 0.15
    test_split: float = 0.15

    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


CFG = CFG()


# ================= UTILS =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_stems(images_dir: str) -> List[str]:
    stems = []
    for f in os.listdir(images_dir):
        if f.lower().endswith(IMG_EXTS):
            stems.append(os.path.splitext(f)[0])
    return sorted(set(stems))


def find_image_path(images_dir: str, stem: str) -> str:
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


# ================= DATASET =================
class EyeSegDataset(Dataset):
    def __init__(self, stems: List[str], images_dir: str, masks_dir: str, img_size: int, augment: bool):
        self.stems = stems
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.stems)

    def _read_pair(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        img_path = find_image_path(self.images_dir, stem)
        if not img_path:
            raise FileNotFoundError(f"Image not found: {stem}")

        mask_path = os.path.join(self.masks_dir, stem + ".png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if bgr is None or mask is None:
            raise ValueError(f"Read error for: {stem}")

        bgr = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        return img, mask

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.3:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if random.random() < 0.3:
            alpha = 0.9 + random.random() * 0.2
            beta = (random.random() - 0.5) * 0.08
            img = np.clip(img * alpha + beta, 0, 1)

        return img, mask

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img, mask = self._read_pair(stem)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return img_t, mask_t, stem


# ================= SPLIT =================
def split_stems(stems, val_split, test_split, seed):
    rng = random.Random(seed)
    stems = stems.copy()
    rng.shuffle(stems)

    n = len(stems)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test = stems[:n_test]
    val = stems[n_test:n_test + n_val]
    train = stems[n_test + n_val:]

    return train, val, test


# ================= MODEL =================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()

        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        mid = self.mid(self.pool4(d4))

        u4 = self.up4(mid)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)


# ================= LOSS =================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return bce + dice_loss


# ================= TRAIN =================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    losses = []
    dices = []
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())

        p = (torch.sigmoid(out) > 0.5).float()
        inter = (p * y).sum((2, 3))
        union = p.sum((2, 3)) + y.sum((2, 3))
        dice = ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        dices.append(dice.item())

    return float(np.mean(losses)), float(np.mean(dices))


def main():
    seed_everything(CFG.seed)
    device = get_device()
    print("Device:", device)

    stems = list_stems(CFG.images_dir)
    stems = [s for s in stems if os.path.exists(os.path.join(CFG.masks_dir, s + ".png"))]

    if not stems:
        raise RuntimeError("Нет пар image+mask")

    train_s, val_s, test_s = split_stems(stems, CFG.val_split, CFG.test_split, CFG.seed)

    train_ds = EyeSegDataset(train_s, CFG.images_dir, CFG.masks_dir, CFG.img_size, augment=True)
    val_ds = EyeSegDataset(val_s, CFG.images_dir, CFG.masks_dir, CFG.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

    model = UNet().to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    os.makedirs(CFG.out_dir, exist_ok=True)
    best_val = 1e9
    best_path = os.path.join(CFG.out_dir, CFG.model_name)

    for e in range(1, CFG.epochs + 1):
        tr = train_epoch(model, train_loader, opt, loss_fn, device)
        val, dice = eval_epoch(model, val_loader, loss_fn, device)

        print(f"[{e:02d}/{CFG.epochs}] train={tr:.4f} val={val:.4f} dice={dice:.4f}")

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), best_path)
            print("  ✅ saved:", best_path)

    print("\nTraining done.")
    print("Best model:", best_path)
    print("Next: test evaluation + prediction visualization.")


if __name__ == "__main__":
    main()
