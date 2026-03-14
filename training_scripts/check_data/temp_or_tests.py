import os
import shutil

IMAGES_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
MASKS_DIR  = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks"
EXTRA_DIR  = os.path.join(MASKS_DIR, "_extra")

os.makedirs(EXTRA_DIR, exist_ok=True)

img_stems = set()
for fn in os.listdir(IMAGES_DIR):
    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img_stems.add(os.path.splitext(fn)[0])

moved = 0
for fn in os.listdir(MASKS_DIR):
    if not fn.lower().endswith(".png"):
        continue
    stem = os.path.splitext(fn)[0]
    if stem not in img_stems:
        shutil.move(os.path.join(MASKS_DIR, fn), os.path.join(EXTRA_DIR, fn))
        moved += 1

print("Images:", len(img_stems))
print("Masks left:", len([f for f in os.listdir(MASKS_DIR) if f.lower().endswith('.png')]))
print("Moved to _extra:", moved)
print("Extra folder:", EXTRA_DIR)
