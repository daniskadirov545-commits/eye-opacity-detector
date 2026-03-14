import os
import cv2
import numpy as np
from pathlib import Path

# ================== PATHS ==================
IMAGES_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
OUT_DIR    = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks_auto"

# ================== SETTINGS ==================
# Можно менять под датасет:
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Насколько "белёсое" (чем меньше, тем строже)
AB_CLOSE_THR = 16  # 10..28 (строже -> меньше ложных)

# Насколько "яркое" (k больше -> строже, меньше масок)
K = 0.65           # 0.45..0.90 (меньше -> больше выделяет)

# Минимальная площадь компоненты (пиксели)
MIN_AREA = 80      # 30..200

# Морфология
OPEN_K = 3         # 0 отключить, 3/5/7
CLOSE_K = 5        # 0 отключить, 3/5/7
DILATE_ITERS = 1   # 0..2

# Hough ROI (если круг не находится, ROI = весь кадр)
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 80
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 28  # меньше -> чаще находит круг (но ложные)
HOUGH_MIN_R_FRAC = 0.18
HOUGH_MAX_R_FRAC = 0.48

OVERWRITE = False  # True = перегенерить даже если маска уже есть

# ================== HELPERS ==================
def list_image_stems(images_dir: str):
    stems = []
    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
            stems.append(Path(f).stem)
    return sorted(set(stems))

def find_image_for_stem(stem: str, images_dir: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def make_circular_roi(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    min_r = int(min(h, w) * HOUGH_MIN_R_FRAC)
    max_r = int(min(h, w) * HOUGH_MAX_R_FRAC)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r
    )

    roi = np.zeros((h, w), dtype=np.uint8)

    if circles is None:
        roi[:] = 255
        return roi

    circles = np.uint16(np.around(circles))
    # берём самый большой круг (обычно это роговица/радужка)
    circles = sorted(circles[0], key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    # чуть уменьшим радиус, чтобы меньше цеплять склеру/веки
    r = int(r * 0.92)
    cv2.circle(roi, (int(x), int(y)), max(10, r), 255, -1)
    return roi

def clahe_if_needed(gray: np.ndarray):
    if not USE_CLAHE:
        return gray
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)

def connected_filter(mask255: np.ndarray, min_area: int):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask255, connectivity=8)
    out = np.zeros_like(mask255)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def auto_mask_for_image(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    roi = make_circular_roi(bgr)

    # LAB: L - яркость, a/b - цвет
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    L2 = clahe_if_needed(L)

    # "белёсость": a и b близко к 128 (нейтральный)
    ab_close = (np.abs(a.astype(np.int16) - 128) <= AB_CLOSE_THR) & (np.abs(b.astype(np.int16) - 128) <= AB_CLOSE_THR)

    # динамический порог яркости по ROI
    L_roi = L2[roi > 0]
    if L_roi.size < 50:
        thr_L = 200
    else:
        # чем меньше K, тем легче пройти порог
        thr_L = int(np.clip(np.percentile(L_roi, 85) + K * (np.percentile(L_roi, 95) - np.percentile(L_roi, 50)), 120, 245))

    bright = (L2 >= thr_L)

    m = (bright & ab_close & (roi > 0)).astype(np.uint8) * 255

    # морфология
    if OPEN_K and OPEN_K >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    if CLOSE_K and CLOSE_K >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if DILATE_ITERS > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.dilate(m, k, iterations=DILATE_ITERS)

    # фильтр по площади
    m = connected_filter(m, MIN_AREA)

    return m

# ================== MAIN ==================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    stems = list_image_stems(IMAGES_DIR)
    if not stems:
        print("Нет изображений в:", IMAGES_DIR)
        return

    made = 0
    skipped = 0
    errors = 0

    for i, stem in enumerate(stems, 1):
        img_path = find_image_for_stem(stem, IMAGES_DIR)
        if not img_path:
            continue

        out_path = os.path.join(OUT_DIR, stem + ".png")
        if (not OVERWRITE) and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise ValueError("cv2.imread вернул None")

            m = auto_mask_for_image(bgr)
            cv2.imwrite(out_path, m)
            made += 1

            if i % 25 == 0 or i == len(stems):
                print(f"[{i}/{len(stems)}] saved={made} skipped={skipped} errors={errors}")

        except Exception as e:
            errors += 1
            print("ERR:", stem, "->", e)

    print("\nDone.")
    print("Saved:", made)
    print("Skipped:", skipped)
    print("Errors:", errors)
    print("Out dir:", OUT_DIR)

if __name__ == "__main__":
    main()