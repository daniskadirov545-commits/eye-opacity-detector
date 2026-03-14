import os
import time
import cv2
import numpy as np

# === PATHS ===
IMAGES_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\images_all"
MASKS_AUTO_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks_auto"
MASKS_EDIT_DIR = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\masks"

# === SETTINGS ===
WINDOW_NAME = "Mask Editor: Original | Mask | Overlay"
OVERLAY_ALPHA = 0.45
MAX_TOTAL_WIDTH = 1500

DEFAULT_BRUSH = 18
MIN_BRUSH = 2
MAX_BRUSH = 120

# Ограничение частоты перерисовки (FPS)
DRAW_FPS = 30

# ========== STATE ==========
brush = DEFAULT_BRUSH
dirty = False
show_overlay = True

drawing = False
erasing = False

cur_img_bgr = None
cur_mask = None
cur_img_path = ""
cur_mask_path = ""
cur_stem = ""
mask_source = "empty"

scale = 1.0

# для ускоренного рисования
last_pt = None

# throttle redraw
LAST_DRAW = 0.0

# Image counter state
current_image_idx = 0
images = []
total_images = 0

# ----------------- HELPERS -----------------

def find_image_for_stem(stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


def ensure_mask_from_sources(stem, shape_hw):
    edited_path = os.path.join(MASKS_EDIT_DIR, stem + ".png")
    auto_path = os.path.join(MASKS_AUTO_DIR, stem + ".png")

    m = None
    source = "empty"

    if os.path.exists(edited_path):
        m = cv2.imread(edited_path, cv2.IMREAD_GRAYSCALE)
        source = "edited"
    elif os.path.exists(auto_path):
        m = cv2.imread(auto_path, cv2.IMREAD_GRAYSCALE)
        source = "auto"

    if m is None:
        m = np.zeros(shape_hw, dtype=np.uint8)
        source = "empty"

    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    m = (m > 127).astype(np.uint8) * 255
    return m, source


def reset_to_auto(stem, shape_hw):
    auto_path = os.path.join(MASKS_AUTO_DIR, stem + ".png")
    if not os.path.exists(auto_path):
        return None

    m = cv2.imread(auto_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None

    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    m = (m > 127).astype(np.uint8) * 255
    return m


def mask_to_overlay(bgr, mask):
    overlay = bgr.copy()
    overlay[mask > 127] = (0, 0, 255)
    return cv2.addWeighted(overlay, OVERLAY_ALPHA, bgr, 1 - OVERLAY_ALPHA, 0)


def resize_to_fit_width(img, max_w):
    global scale
    h, w = img.shape[:2]
    if w <= max_w:
        scale = 1.0
        return img
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_image_coords(x, y):
    if scale == 1.0:
        return x, y
    return int(x / scale), int(y / scale)


def brush_change(delta):
    global brush
    brush = int(np.clip(brush + delta, MIN_BRUSH, MAX_BRUSH))


def save_mask():
    global dirty, mask_source
    os.makedirs(os.path.dirname(cur_mask_path), exist_ok=True)
    cv2.imwrite(cur_mask_path, cur_mask)
    dirty = False
    mask_source = "edited"
    print("Saved:", cur_mask_path)


def ask_save_if_dirty():
    if dirty:
        save_mask()


def delete_mask():
    global dirty, mask_source
    if os.path.exists(cur_mask_path):
        os.remove(cur_mask_path)  # Удаление маски
        print(f"Deleted mask: {cur_mask_path}")

    img_path = cur_img_path
    if os.path.exists(img_path):
        os.remove(img_path)  # Удаление изображения
        print(f"Deleted image: {img_path}")

    # Очистить текущие данные
    cur_mask = np.zeros_like(cur_mask)
    cur_img_bgr = None

    dirty = False
    mask_source = "empty"
    # Перейти к следующему изображению
    go_to_next_image()


def go_to_next_image():
    global current_image_idx
    current_image_idx = (current_image_idx + 1) % len(images)  # Переход к следующему изображению
    load_pair(images[current_image_idx])
    redraw_throttled(force=True)


def go_to_previous_image():
    global current_image_idx
    current_image_idx = (current_image_idx - 1) % len(images)  # Переход к предыдущему изображению
    load_pair(images[current_image_idx])
    redraw_throttled(force=True)


def load_pair(stem):
    global cur_img_bgr, cur_mask, cur_img_path, cur_mask_path, cur_stem, dirty, mask_source, last_pt

    img_path = find_image_for_stem(stem)
    if not img_path:
        raise FileNotFoundError(f"No image for {stem}")

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Cannot read {img_path}")

    mask, source = ensure_mask_from_sources(stem, bgr.shape[:2])

    cur_img_bgr = bgr
    cur_mask = mask
    cur_img_path = img_path
    cur_mask_path = os.path.join(MASKS_EDIT_DIR, stem + ".png")
    cur_stem = stem
    dirty = False
    mask_source = source
    last_pt = None

    # Update the counter
    print(f"Image {current_image_idx + 1}/{total_images}")


# ----------------- RENDER -----------------

def make_triptych(bgr, mask):
    h, w = bgr.shape[:2]
    header_h = 60

    original = bgr.copy()
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = mask_to_overlay(bgr, mask) if show_overlay else bgr.copy()

    cv2.putText(original, "Original", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(mask3, "Mask (edit)", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(overlay, "Overlay" if show_overlay else "Overlay OFF", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    trip = np.hstack([original, mask3, overlay])

    header = np.zeros((header_h, w * 3, 3), dtype=np.uint8)
    name = os.path.basename(cur_img_path)
    status = "DIRTY" if dirty else "SAVED"
    cv2.putText(header,
                f"{name} | source={mask_source} | brush={brush}px | {status}",
                (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return np.vstack([header, trip])


def redraw():
    view = make_triptych(cur_img_bgr, cur_mask)
    view_disp = resize_to_fit_width(view, MAX_TOTAL_WIDTH)
    cv2.imshow(WINDOW_NAME, view_disp)


def redraw_throttled(force=False):
    global LAST_DRAW
    now = time.time()
    if force or (now - LAST_DRAW >= 1.0 / DRAW_FPS):
        redraw()
        LAST_DRAW = now


# ----------------- MOUSE -----------------

def on_mouse(event, x, y, flags, param):
    global drawing, erasing, dirty, last_pt

    if cur_mask is None or cur_img_bgr is None:
        return

    x0, y0 = to_image_coords(x, y)

    h, w = cur_img_bgr.shape[:2]
    header_h = 60

    if y0 < header_h:
        return

    y_img = y0 - header_h

    # редактируем только Mask/Overlay (2-я и 3-я колонки)
    if x0 < w:
        return

    x_in = x0 % w
    y_in = y_img

    if not (0 <= y_in < h):
        return

    # толщину линии берём больше радиуса, чтобы кисть ощущалась как круглая
    thickness = max(1, brush * 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        erasing = False
        last_pt = (x_in, y_in)
        cv2.circle(cur_mask, last_pt, brush, 255, -1)
        dirty = True
        redraw_throttled(force=True)

    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        drawing = False
        last_pt = (x_in, y_in)
        cv2.circle(cur_mask, last_pt, brush, 0, -1)
        dirty = True
        redraw_throttled(force=True)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and last_pt is not None:
            cv2.line(cur_mask, last_pt, (x_in, y_in), 255, thickness=thickness)
            last_pt = (x_in, y_in)
            dirty = True
            redraw_throttled()

        elif erasing and last_pt is not None:
            cv2.line(cur_mask, last_pt, (x_in, y_in), 0, thickness=thickness)
            last_pt = (x_in, y_in)
            dirty = True
            redraw_throttled()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_pt = None
        redraw_throttled(force=True)

    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False
        last_pt = None
        redraw_throttled(force=True)


# ----------------- MAIN -----------------

def main():
    global total_images, images, current_image_idx

    stems = []
    for f in os.listdir(IMAGES_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            stems.append(os.path.splitext(f)[0])
    stems = sorted(set(stems))

    images = stems
    total_images = len(images)

    if not images:
        print("No images found in IMAGES_DIR.")
        return

    os.makedirs(MASKS_EDIT_DIR, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    current_image_idx = 0
    load_pair(images[current_image_idx])
    redraw_throttled(force=True)

    print("Controls:")
    print(" LMB on Mask/Overlay - draw (fast stroke)")
    print(" RMB on Mask/Overlay - erase (fast stroke)")
    print(" S - save")
    print(" D / L / ->  next (auto-save if dirty)")
    print(" A / J / <-  prev (auto-save if dirty)")
    print(" + - or ] [  brush size")
    print(" O - toggle overlay")
    print(" R - reset to auto mask")
    print(" DEL - delete mask and image")
    print(" Q or ESC - quit (auto-save if dirty)")
    print(f" Redraw FPS limit: {DRAW_FPS}")

    KEY_LEFT = 2424832
    KEY_RIGHT = 2555904
    KEY_DELETE = 65535  # Delete key

    global show_overlay, dirty, mask_source

    while True:
        key = cv2.waitKeyEx(0)

        if key in (ord('q'), ord('Q'), 27):
            ask_save_if_dirty()
            break

        elif key in (ord('s'), ord('S')):
            save_mask()
            redraw_throttled(force=True)

        elif key in (ord('r'), ord('R')):
            auto = reset_to_auto(cur_stem, cur_img_bgr.shape[:2])
            if auto is not None:
                cur_mask[:] = auto
                dirty = True
                mask_source = "auto"
                print("Reset to AUTO mask")
            else:
                print("No auto mask for this image")
            redraw_throttled(force=True)

        elif key in (ord('d'), ord('D'), ord('l'), ord('L'), KEY_RIGHT):
            ask_save_if_dirty()
            go_to_next_image()

        elif key in (ord('a'), ord('A'), ord('j'), ord('J'), KEY_LEFT):
            ask_save_if_dirty()
            go_to_previous_image()

        elif key in (ord('+'), ord('='), ord(']')):
            brush_change(2)
            redraw_throttled(force=True)

        elif key in (ord('-'), ord('_'), ord('[')):
            brush_change(-2)
            redraw_throttled(force=True)

        elif key in (ord('o'), ord('O')):
            show_overlay = not show_overlay
            redraw_throttled(force=True)

        elif key == KEY_DELETE:
            delete_mask()

        else:
            redraw_throttled(force=True)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
