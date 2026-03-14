import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn as nn
import sys, os

# для билда
def resource_path(rel):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.abspath("."), rel)


model_path = resource_path("unet_corneal_opacity.pt")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet_corneal_opacity.pt")
UNET_INPUT_SIZE = 256

P95_SWITCH = 0.65
THR_STRICT = 0.88
THR_SENSITIVE = 0.68
THR_CLAMP_MIN = 0.45
THR_CLAMP_MAX = 0.90

OVERLAY_ALPHA = 0.45

MIN_AREA_STRICT = 800
MIN_AREA_SENSITIVE = 120
NOISE_FLOOR_PERCENT = 3.0

OPEN_KERNEL = (3, 3)
CLOSE_KERNEL = (5, 5)
DILATE_KERNEL = (3, 3)
DILATE_ITERS = 1

CENTER_CUT_ENABLE = True
CENTER_CUT_RADIUS_STRICT = 0.30


class ToolTip:
    def __init__(self, widget, text, wraplength=360):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwin = None

        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<Button-1>", self.show)

    def show(self, event=None):
        if self.tipwin is not None:
            return

        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        y = self.widget.winfo_rooty()

        self.tipwin = tk.Toplevel(self.widget)
        self.tipwin.wm_overrideredirect(True)
        self.tipwin.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tipwin,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10),
            wraplength=self.wraplength,
        )
        label.pack(ipadx=8, ipady=6)

    def hide(self, event=None):
        if self.tipwin is not None:
            self.tipwin.destroy()
            self.tipwin = None


# Работа с изображением

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def overlay_mask_bgr(bgr, mask255, alpha=OVERLAY_ALPHA):
    result = bgr.copy()
    result[mask255 == 255] = (0, 0, 255)
    return cv2.addWeighted(result, alpha, bgr, 1 - alpha, 0)


def roi_circle_to_mask(height, width, circle):
    if circle is None:
        return np.full((height, width), 255, dtype=np.uint8)

    cx, cy, radius = circle
    cx = int(np.clip(cx, 0, width - 1))
    cy = int(np.clip(cy, 0, height - 1))
    radius = int(max(5, radius))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def draw_roi_circle(bgr, circle):
    image = bgr.copy()
    if circle is None:
        return image

    cx, cy, radius = circle
    center = (int(cx), int(cy))
    radius = int(radius)

    cv2.circle(image, center, radius, (0, 255, 255), 2)
    cv2.circle(image, center, 2, (0, 255, 255), -1)
    return image


def compute_percent_from_mask(mask255, roi_mask255):
    clean_mask = (mask255 > 127).astype(np.uint8) * 255
    clean_mask = cv2.bitwise_and(clean_mask, roi_mask255)

    opacity_area = int(np.sum(clean_mask == 255))
    roi_area = int(np.sum(roi_mask255 == 255))
    percent = 100.0 * opacity_area / max(roi_area, 1)
    return percent, clean_mask


def keep_large_components(mask255, min_area):
    binary = (mask255 > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            filtered[labels == index] = 1

    return filtered.astype(np.uint8) * 255


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

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def load_unet_model(model_path):
    if not os.path.exists(model_path):
        return None, None, f"Файл модели не найден: {model_path}"

    device = pick_device()
    model = UNet().to(device)

    try:
        model = UNet()
        state = torch.load("unet_corneal_opacity.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device, f"U-Net загружен. Device: {device}"
    except Exception as error:
        return None, None, f"Ошибка загрузки U-Net: {error}"

@torch.no_grad()
def unet_predict_mask_255(model, device, bgr, roi_mask255):
    height, width = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (UNET_INPUT_SIZE, UNET_INPUT_SIZE), interpolation=cv2.INTER_AREA)

    x = small.astype(np.float32) / 255.0
    x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(x_tensor)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    p95 = float(np.percentile(probs, 95))
    threshold = THR_STRICT if p95 < P95_SWITCH else THR_SENSITIVE
    threshold = float(np.clip(threshold, THR_CLAMP_MIN, THR_CLAMP_MAX))

    mask_small = (probs >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.bitwise_and(mask, roi_mask255)

    if CENTER_CUT_ENABLE and p95 < P95_SWITCH:
        inner = np.zeros_like(mask)
        center_x = width // 2
        center_y = height // 2
        inner_radius = int(CENTER_CUT_RADIUS_STRICT * min(height, width))
        cv2.circle(inner, (center_x, center_y), inner_radius, 255, -1)
        mask[inner == 255] = 0
        mask = cv2.bitwise_and(mask, roi_mask255)

    min_area = MIN_AREA_STRICT if p95 < P95_SWITCH else MIN_AREA_SENSITIVE
    mask = keep_large_components(mask, min_area=min_area)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_KERNEL)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.dilate(mask, dilate_kernel, iterations=DILATE_ITERS)
    mask = cv2.bitwise_and(mask, roi_mask255)

    return mask, threshold, p95, min_area


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)

        self.box = ttk.LabelFrame(self, text=title, padding=6)
        self.box.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.box, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.canvas.pack(fill="both", expand=True)

        self.pil_original = None
        self.bgr_original = None
        self.pil_shown = None
        self.tk_img = None

        self.display_scale = 1.0
        self.display_offset = (0, 0)

        self.roi_circle_orig = None

        self._dragging = False
        self._center_orig = None
        self._preview_circle_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Configure>", lambda event: self.redraw())

    def set_image_pil(self, pil_img):
        self.pil_original = pil_img.copy().convert("RGB")
        self.bgr_original = pil_to_bgr(self.pil_original)
        self.pil_shown = self.pil_original.copy()

        self.roi_circle_orig = None
        self._center_orig = None
        self._dragging = False
        self._preview_circle_id = None

        self.redraw()

    def show_bgr(self, bgr_to_show):
        self.pil_shown = bgr_to_pil(bgr_to_show)
        self.redraw()

    def reset_view_to_original(self):
        if self.pil_original is not None:
            self.pil_shown = self.pil_original.copy()
            self.redraw()

    def reset_roi(self):
        self.roi_circle_orig = None
        self._center_orig = None
        self._dragging = False
        self._preview_circle_id = None
        self.redraw()

    def get_roi_mask(self):
        if self.bgr_original is None:
            return None

        height, width = self.bgr_original.shape[:2]
        return roi_circle_to_mask(height, width, self.roi_circle_orig)

    def redraw(self):
        self.canvas.delete("all")

        if self.pil_shown is None:
            self.canvas.create_text(10, 10, anchor="nw", text="Снимок не загружен", fill="#666")
            return

        canvas_width = max(self.canvas.winfo_width(), 20)
        canvas_height = max(self.canvas.winfo_height(), 20)

        original_width, original_height = self.pil_shown.size
        scale = min(canvas_width / original_width, canvas_height / original_height, 1.0)
        self.display_scale = scale

        shown_width = int(original_width * scale)
        shown_height = int(original_height * scale)
        shown_image = self.pil_shown.resize((shown_width, shown_height), Image.Resampling.LANCZOS)

        offset_x = (canvas_width - shown_width) // 2
        offset_y = (canvas_height - shown_height) // 2
        self.display_offset = (offset_x, offset_y)

        self.tk_img = ImageTk.PhotoImage(shown_image)
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_img)

        if self.roi_circle_orig is not None:
            cx, cy, radius = self.roi_circle_orig
            disp_x, disp_y = self.orig_to_disp(cx, cy)
            disp_radius = int(radius * self.display_scale)
            self.canvas.create_oval(
                disp_x - disp_radius,
                disp_y - disp_radius,
                disp_x + disp_radius,
                disp_y + disp_radius,
                outline="yellow",
                width=3,
            )
            self.canvas.create_oval(disp_x - 2, disp_y - 2, disp_x + 2, disp_y + 2, fill="yellow", outline="")

    def orig_to_disp(self, x, y):
        offset_x, offset_y = self.display_offset
        return offset_x + int(x * self.display_scale), offset_y + int(y * self.display_scale)

    def disp_to_orig(self, x, y):
        offset_x, offset_y = self.display_offset
        x = (x - offset_x) / max(self.display_scale, 1e-9)
        y = (y - offset_y) / max(self.display_scale, 1e-9)

        if self.bgr_original is not None:
            height, width = self.bgr_original.shape[:2]
            x = float(np.clip(x, 0, width - 1))
            y = float(np.clip(y, 0, height - 1))

        return int(x), int(y)

    def on_down(self, event):
        if self.bgr_original is None:
            return

        self._center_orig = self.disp_to_orig(event.x, event.y)
        self._dragging = True

        if self._preview_circle_id is not None:
            self.canvas.delete(self._preview_circle_id)
            self._preview_circle_id = None

    def on_drag(self, event):
        if not self._dragging or self.bgr_original is None or self._center_orig is None:
            return

        center_x, center_y = self._center_orig
        point_x, point_y = self.disp_to_orig(event.x, event.y)
        radius = int(((point_x - center_x) ** 2 + (point_y - center_y) ** 2) ** 0.5)

        disp_x, disp_y = self.orig_to_disp(center_x, center_y)
        disp_radius = int(radius * self.display_scale)

        if self._preview_circle_id is not None:
            self.canvas.delete(self._preview_circle_id)

        self._preview_circle_id = self.canvas.create_oval(
            disp_x - disp_radius,
            disp_y - disp_radius,
            disp_x + disp_radius,
            disp_y + disp_radius,
            outline="yellow",
            width=3,
        )

    def on_up(self, event):
        if not self._dragging or self.bgr_original is None or self._center_orig is None:
            return

        center_x, center_y = self._center_orig
        point_x, point_y = self.disp_to_orig(event.x, event.y)
        radius = int(((point_x - center_x) ** 2 + (point_y - center_y) ** 2) ** 0.5)

        self._dragging = False
        self._center_orig = None
        self._preview_circle_id = None

        if radius < 10:
            self.roi_circle_orig = None
        else:
            self.roi_circle_orig = (center_x, center_y, radius)

        self.redraw()


class EyeComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Помутнение роговицы — U-Net + круглый ROI")
        self.root.geometry("1750x960")

        self.single_mode = tk.BooleanVar(value=False)

        self.before_pil = None
        self.after_pil = None

        self.unet_model, self.unet_device, self.unet_status_msg = load_unet_model(MODEL_PATH)

        self.create_widgets()
        self.update_unet_status()
        self.update_mode()

        if self.unet_model is None:
            messagebox.showwarning("U-Net", self.unet_status_msg)

    def create_widgets(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        self.instruction_label = ttk.Label(top, font=("Arial", 13, "bold"))
        self.instruction_label.pack(side="left", padx=(0, 12))

        self.btn_before = ttk.Button(top, text="До операции", command=lambda: self.load_image("before"))
        self.btn_before.pack(side="left", padx=5)

        self.btn_after = ttk.Button(top, text="После операции", command=lambda: self.load_image("after"))
        self.btn_after.pack(side="left", padx=5)

        self.btn_run = ttk.Button(top, text="Сравнить", command=self.run_analysis)
        self.btn_run.pack(side="left", padx=10)

        self.mode_check = ttk.Checkbutton(
            top,
            text="Одиночный анализ",
            variable=self.single_mode,
            command=self.update_mode,
        )
        self.mode_check.pack(side="right", padx=10)

        settings_box = ttk.LabelFrame(top, text="Настройки", padding=8)
        settings_box.pack(side="right", padx=5)
        self.unet_info_label = ttk.Label(settings_box, text="Модель: проверка...")
        self.unet_info_label.grid(row=0, column=0, columnspan=2, sticky="w")
        ToolTip(
            self.unet_info_label,
            "Приложение использует только U-Net модель из файла unet_corneal_opacity.pt рядом со скриптом.",
        )
        ttk.Button(settings_box, text="Сброс ROI", command=self.reset_roi).grid(
            row=1, column=0, sticky="we", pady=(6, 0)
        )
        ttk.Button(settings_box, text="Показать исходник", command=self.reset_view).grid(
            row=1, column=1, sticky="we", pady=(6, 0)
        )

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.before_panel = ImagePanel(left, "До операции")
        self.before_panel.pack(fill="both", expand=True, pady=5)

        self.after_panel = ImagePanel(left, "После операции")
        self.after_panel.pack(fill="both", expand=True, pady=5)

        right = ttk.Frame(main)
        right.pack(side="right", fill="y")

        results_box = ttk.LabelFrame(right, text="Результаты анализа", padding=8)
        results_box.pack(fill="x")

        self.results_text = tk.Text(results_box, width=48, height=28, wrap="word")
        self.results_text.pack()

    def update_unet_status(self):
        if self.unet_model is not None:
            self.unet_info_label.config(text=f"Модель: подключена ({self.unet_device})")
        else:
            self.unet_info_label.config(text="Модель: не загружена")

    def update_mode(self):
        if self.single_mode.get():
            self.after_panel.pack_forget()
            self.btn_after.config(state="disabled")
            self.btn_before.config(text="Снимок")
            self.btn_run.config(text="Анализировать")
            self.instruction_label.config(
                text="Загрузите снимок и выделите круг мышкой: клик — центр, протяжка — радиус"
            )
        else:
            if not self.after_panel.winfo_ismapped():
                self.after_panel.pack(fill="both", expand=True, pady=5)

            self.btn_after.config(state="normal")
            self.btn_before.config(text="До операции")
            self.btn_run.config(text="Сравнить")
            self.instruction_label.config(
                text="Загрузите ДО и ПОСЛЕ, при необходимости выделите ROI и нажмите «Сравнить»"
            )

        self.results_text.delete("1.0", "end")

    def reset_roi(self):
        self.before_panel.reset_roi()
        self.after_panel.reset_roi()

    def reset_view(self):
        self.before_panel.reset_view_to_original()
        self.after_panel.reset_view_to_original()

    def load_image(self, which):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not path:
            return

        image = Image.open(path).convert("RGB")

        if which == "before":
            self.before_pil = image
            self.before_panel.set_image_pil(image)
        else:
            self.after_pil = image
            self.after_panel.set_image_pil(image)

    def run_analysis(self):
        try:
            if self.single_mode.get():
                self.run_single()
            else:
                self.run_compare()
        except Exception as error:
            import traceback
            messagebox.showerror("Ошибка при анализе", f"{error}\n\n{traceback.format_exc()}")

    def analyze_one(self, bgr, roi_mask255):
        if self.unet_model is None:
            raise RuntimeError("Модель U-Net не загружена. Проверьте файл unet_corneal_opacity.pt рядом со скриптом.")

        mask255, threshold, p95, min_area = unet_predict_mask_255(
            self.unet_model,
            self.unet_device,
            bgr,
            roi_mask255,
        )
        percent, mask_roi = compute_percent_from_mask(mask255, roi_mask255)

        if percent < NOISE_FLOOR_PERCENT:
            percent = 0.0
            mask_roi[:] = 0

        info = f"U-Net (thr={threshold:.2f}, p95={p95:.2f}, min_area={min_area})"
        return percent, mask_roi, info

    def run_single(self):
        if self.before_pil is None:
            messagebox.showwarning("Ошибка", "Загрузите снимок")
            return

        bgr = pil_to_bgr(self.before_pil)
        roi_mask = self.before_panel.get_roi_mask()
        roi_circle = self.before_panel.roi_circle_orig

        percent, mask_roi, method_info = self.analyze_one(bgr, roi_mask)

        shown = overlay_mask_bgr(bgr, mask_roi, alpha=OVERLAY_ALPHA)
        shown = draw_roi_circle(shown, roi_circle)
        self.before_panel.show_bgr(shown)

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"Метод: {method_info}\n")
        self.results_text.insert("end", f"Модель: {MODEL_PATH}\n")
        self.results_text.insert(
            "end",
            f"ROI: {'выбран' if roi_circle else 'не выбран (анализ всей картинки)'}\n",
        )
        self.results_text.insert("end", f"Шум-флор: < {NOISE_FLOOR_PERCENT:.1f}% -> 0%\n\n")
        self.results_text.insert("end", f"ПОМУТНЕНИЕ: {percent:.2f}%\n")

    def run_compare(self):
        if self.before_pil is None or self.after_pil is None:
            messagebox.showwarning("Ошибка", "Загрузите оба снимка")
            return

        before_bgr = pil_to_bgr(self.before_pil)
        after_bgr = pil_to_bgr(self.after_pil)

        before_roi_mask = self.before_panel.get_roi_mask()
        after_roi_mask = self.after_panel.get_roi_mask()

        before_roi_circle = self.before_panel.roi_circle_orig
        after_roi_circle = self.after_panel.roi_circle_orig

        before_percent, before_mask, before_info = self.analyze_one(before_bgr, before_roi_mask)
        after_percent, after_mask, after_info = self.analyze_one(after_bgr, after_roi_mask)
        diff = before_percent - after_percent

        shown_before = overlay_mask_bgr(before_bgr, before_mask, alpha=OVERLAY_ALPHA)
        shown_before = draw_roi_circle(shown_before, before_roi_circle)
        self.before_panel.show_bgr(shown_before)

        shown_after = overlay_mask_bgr(after_bgr, after_mask, alpha=OVERLAY_ALPHA)
        shown_after = draw_roi_circle(shown_after, after_roi_circle)
        self.after_panel.show_bgr(shown_after)

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"ДО: {before_info}\n")
        self.results_text.insert("end", f"ПОСЛЕ: {after_info}\n")
        self.results_text.insert("end", f"Модель: {MODEL_PATH}\n")
        self.results_text.insert("end", f"Шум-флор: < {NOISE_FLOOR_PERCENT:.1f}% -> 0%\n\n")
        self.results_text.insert("end", f"ROI ДО: {'выбран' if before_roi_circle else 'вся картинка'}\n")
        self.results_text.insert("end", f"ROI ПОСЛЕ: {'выбран' if after_roi_circle else 'вся картинка'}\n\n")
        self.results_text.insert("end", f"ДО: {before_percent:.2f}%\n")
        self.results_text.insert("end", f"ПОСЛЕ: {after_percent:.2f}%\n")
        self.results_text.insert("end", f"Разница: {diff:.2f}%\n")


root = tk.Tk()
app = EyeComparisonApp(root)
root.mainloop()
