import os
import cv2
import numpy as np

IMAGES_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
MASKS_DIR  = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
os.makedirs(MASKS_DIR, exist_ok=True)

created = 0
skipped = 0

for f in os.listdir(IMAGES_DIR):
    if not f.lower().endswith(IMG_EXTS):
        continue
    stem = os.path.splitext(f)[0]
    mask_path = os.path.join(MASKS_DIR, stem + ".png")
    if os.path.exists(mask_path):
        skipped += 1
        continue

    img = cv2.imread(os.path.join(IMAGES_DIR, f))
    if img is None:
        continue

    h, w = img.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(mask_path, empty)
    created += 1

print("Пустые маски созданы:", created)
print("Уже были маски:", skipped)
