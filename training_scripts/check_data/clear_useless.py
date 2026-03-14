import os
import shutil

IMAGES_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
MASKS_DIR  = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks"
EXTRA_DIR  = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\_extra"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

os.makedirs(EXTRA_DIR, exist_ok=True)

moved = 0
kept = 0

# Собираем список всех stem'ов изображений
image_stems = set()

for f in os.listdir(IMAGES_DIR):
    if f.lower().endswith(IMG_EXTS):
        stem = os.path.splitext(f)[0]
        image_stems.add(stem)

# Проверяем маски
for f in os.listdir(MASKS_DIR):
    if not f.lower().endswith(".png"):
        continue

    stem = os.path.splitext(f)[0]

    if stem not in image_stems:
        shutil.move(
            os.path.join(MASKS_DIR, f),
            os.path.join(EXTRA_DIR, f)
        )
        moved += 1
    else:
        kept += 1

print("Оставлено (есть изображение):", kept)
print("Перемещено в _extra (нет пары):", moved)
