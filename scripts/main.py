import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn


#Поиск файлов и базовые настройки нейросети
def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)
# Ищем модель в нескольких местах, чтобы программа не падала из-за пути
def find_model_path() -> str:
    candidates = [resource_path("unet_corneal_opacity.pt"),os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "unet_corneal_opacity.pt"),os.path.join(os.getcwd(), "unet_corneal_opacity.pt"),]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    return os.path.abspath(candidates[0])
# Параметры работы сети (взяты вместе с архитектрурой с открытых источников)
MODEL_PATH = find_model_path()
UNET_INPUT_SIZE = 256  # Сеть обучена на картинках 256x256

# Порог подбирается по уверенности сети: когда она уверена, режем строже, когда сомневается — даём более чувствительный режим.
P95_SWITCH = 0.65  # Граница "уверенности": выше - уверена, ниже - не уверена
THR_STRICT = 0.65  # Строгий порог
THR_SENSITIVE = 0.45  # Чувствительный порог
THR_CLAMP_MIN = 0.25  # Нижняя граница порога
THR_CLAMP_MAX = 0.90  # Верхняя граница порога
OVERLAY_ALPHA = 0.45  # Прозрачность красной маски

# Мелкие области чаще всего оказываются шумом, поэтому их отсекаем.
MIN_AREA_STRICT = 200  # Когда сеть уверена - удаляем всё мельче 200 пикселей
MIN_AREA_SENSITIVE = 60  # Когда сомневается - оставляем даже мелкое

# Если итог совсем маленький, считаем это шумом, а не реальным помутнением.
NOISE_FLOOR_PERCENT = 1.0
# Морфология помогает пригладить маску и убрать случайные артефакты.
OPEN_KERNEL = (3, 3)
CLOSE_KERNEL = (3, 3)
DILATE_KERNEL = (3, 3)
DILATE_ITERS = 2  # Сколько раз расширять

# Этот блок можно включить, если сеть начинает путать центр глаза с помутнением.
# Сейчас он выключен.
CENTER_CUT_ENABLE = False
CENTER_CUT_RADIUS_STRICT = 0.25  # Радиус выреза как доля от меньшей стороны картинки


# Всплывающие подсказки в интерфейсе
class ToolTip:
    def __init__(self, widget, text: str, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwin = None
        # Подсказка открывается по наведению и по клику.
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<Button-1>", self.show)
    def show(self, event=None):
        if self.tipwin is not None:
            return
        #Небольшое отдельное окно рядом с элементом
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        y = self.widget.winfo_rooty()
        self.tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, font=("Arial", 10),
            wraplength=self.wraplength
        )
        label.pack(ipadx=8, ipady=6)
    def hide(self, event=None):
        if self.tipwin is not None:
            self.tipwin.destroy()
            self.tipwin = None

#Базовые функции для работы с изображениями
#Эти функции используются и в одиночном анализе, и в сравнении двух снимков.
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Перевести PIL-изображение в формат OpenCV (BGR)."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#Перевести изображение из OpenCV обратно в PIL
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
#Накладываем маску
def overlay_mask_bgr(bgr: np.ndarray, mask255: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    overlay = bgr.copy()
    overlay[mask255 == 255] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
def roi_circle_to_mask(h: int, w: int, circle):
    if circle is None:
        return np.ones((h, w), dtype=np.uint8) * 255
    cx, cy, r = circle
    # На всякий случай ограничиваем координаты границами изображения
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    r = int(max(5, r))  # Минимальный радиус 5
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m
def draw_roi_circle(bgr: np.ndarray, circle):
    out = bgr.copy()
    if circle is None:
        return out
    cx, cy, r = circle
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
    cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 255), -1)
    return out
def compute_percent_from_mask(mask255: np.ndarray, roi_mask255: np.ndarray):
    m = (mask255 > 127).astype(np.uint8) * 255
    # Оставляем только пиксели внутри ROI
    m = cv2.bitwise_and(m, roi_mask255)
    opacity_area = int(np.sum(m == 255))
    roi_area = int(np.sum(roi_mask255 == 255))
    percent = 100.0 * opacity_area / max(roi_area, 1)
    return percent, m
def keep_large_components(mask255: np.ndarray, min_area: int) -> np.ndarray:
    m = (mask255 > 127).astype(np.uint8)
    # Ищем связные компоненты и фильтруем их по площади
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == i] = 1
    return out.astype(np.uint8) * 255


#Архитектура U-Net
# Этот блок я не писал с нуля: базовая архитектура и общая схема были взяты и адаптированы из открытых источников под мой проект.

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
    """Классическая U-Net для сегментации."""

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

        # На выходе получаем карту вероятностей для каждого пикселя
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
    """Выбрать устройство: CUDA, DirectML или CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def load_unet_model(model_path: str):
    """Загрузить веса модели и подготовить её к инференсу."""
    if not os.path.exists(model_path):
        return None, None, f"Файл модели не найден: {model_path}"
    device = pick_device()
    model = UNet().to(device)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()  # режим инференса
        return model, device, f"U-Net загружен. Устройство: {device}"
    except Exception as e:
        return None, None, f"Ошибка загрузки U-Net: {e}"


@torch.no_grad()  # здесь градиенты не нужны, мы только предсказываем
def unet_predict_mask_255(model: nn.Module, device: torch.device, bgr: np.ndarray, roi_mask255: np.ndarray):
    """Прогнать одно изображение через U-Net и вернуть итоговую маску."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Сеть обучалась на 256x256, поэтому приводим вход к этому размеру
    small = cv2.resize(rgb, (UNET_INPUT_SIZE, UNET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = (small.astype(np.float32) / 255.0)
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    # Получаем карту вероятностей
    logits = model(x_t)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    # Оцениваем, насколько уверенно сеть вообще видит область
    p95 = float(np.percentile(probs, 95))

    # Порог делаем адаптивным
    thr = THR_SENSITIVE if p95 < P95_SWITCH else THR_STRICT
    thr = float(np.clip(thr, THR_CLAMP_MIN, THR_CLAMP_MAX))

    # Бинаризуем результат и возвращаемся к исходному размеру
    mask_small = (probs >= thr).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # За пределами ROI маска не нужна
    mask = cv2.bitwise_and(mask, roi_mask255)

    # Этот кусок можно включать для дополнительной защиты от ложных срабатываний
    if CENTER_CUT_ENABLE and p95 < P95_SWITCH:
        hh, ww = mask.shape
        cx, cy = ww // 2, hh // 2
        r_inner = int(CENTER_CUT_RADIUS_STRICT * min(hh, ww))
        inner = np.zeros_like(mask)
        cv2.circle(inner, (cx, cy), r_inner, 255, -1)
        mask[inner == 255] = 0
        mask = cv2.bitwise_and(mask, roi_mask255)

    # Финальная чистка маски
    min_area = MIN_AREA_SENSITIVE if p95 < P95_SWITCH else MIN_AREA_STRICT
    mask = keep_large_components(mask, min_area=min_area)

    # Чистим шум и слегка закрываем дырки
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Чуть расширяем маску, чтобы не обрезать край области
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_KERNEL)
    mask = cv2.dilate(mask, k_dil, iterations=DILATE_ITERS)

    # И ещё раз ограничиваем всё ROI
    mask = cv2.bitwise_and(mask, roi_mask255)

    return mask, thr, p95, min_area


# ===== Панель изображения и выбор ROI =====
# загрузка изображения, показ и ручной выбор области интереса.
class ImagePanel(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        # Внешняя рамка блока
        self.box = ttk.LabelFrame(self, text=title, padding=6)
        self.box.pack(fill="both", expand=True)
        # Поле, где показываем изображение
        self.canvas = tk.Canvas(self.box, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.canvas.pack(fill="both", expand=True)
        # Исходное изображение и его копии для показа
        self.pil_original = None
        self.bgr_original = None
        # То, что сейчас реально видно на экране
        self.pil_shown = None
        self.tk_img = None
        # Параметры отображения на Canvas
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        # ROI храним в координатах исходного изображения
        self.roi_circle_orig = None
        # Переменные для ручного рисования круга
        self._dragging = False
        self._center_orig = None
        self._oval_id = None
        # События мыши
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
    def set_image_pil(self, pil_img: Image.Image):
        self.pil_original = pil_img.copy().convert("RGB")
        self.bgr_original = pil_to_bgr(self.pil_original)
        self.pil_shown = self.pil_original.copy()
        # При новой загрузке ROI начинаем заново
        self.roi_circle_orig = None
        self._center_orig = None
        self._dragging = False
        self.redraw()
    def show_bgr(self, bgr_to_show: np.ndarray):
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
        self.redraw()
    def get_roi_mask(self):
        if self.bgr_original is None:
            return None
        h, w = self.bgr_original.shape[:2]
        return roi_circle_to_mask(h, w, self.roi_circle_orig)
    def redraw(self):
        self.canvas.delete("all")
        if self.pil_shown is None:
            self.canvas.create_text(10, 10, anchor="nw", text="Снимок не загружен", fill="#666")
            return
        # Подгоняем изображение под размеры Canvas
        cw = max(self.canvas.winfo_width(), 20)
        ch = max(self.canvas.winfo_height(), 20)
        # Считаем масштаб, чтобы всё помещалось
        ow, oh = self.pil_shown.size
        scale = min(cw / ow, ch / oh)
        scale = min(scale, 1.0)
        self.display_scale = scale
        dw, dh = int(ow * scale), int(oh * scale)
        img_disp = self.pil_shown.resize((dw, dh), Image.Resampling.LANCZOS)
        # Центрируем изображение внутри области показа
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self.display_offset = (ox, oy)
        self.tk_img = ImageTk.PhotoImage(img_disp)
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_img)
        # Рисовка круга поверх ROI
        if self.roi_circle_orig is not None:
            cx, cy, r = self.roi_circle_orig
            dcx, dcy = self.orig_to_disp(cx, cy)
            dr = int(r * self.display_scale)
            self.canvas.create_oval(dcx - dr, dcy - dr, dcx + dr, dcy + dr, outline="yellow", width=3)
            self.canvas.create_oval(dcx - 2, dcy - 2, dcx + 2, dcy + 2, fill="yellow", outline="")
    def orig_to_disp(self, x, y):
        ox, oy = self.display_offset
        return ox + int(x * self.display_scale), oy + int(y * self.display_scale)
    def disp_to_orig(self, x, y):
        ox, oy = self.display_offset
        x = (x - ox) / max(self.display_scale, 1e-9)
        y = (y - oy) / max(self.display_scale, 1e-9)
        if self.bgr_original is not None:
            h, w = self.bgr_original.shape[:2]
            x = float(np.clip(x, 0, w - 1))
            y = float(np.clip(y, 0, h - 1))
        return int(x), int(y)
    #Запомнили центр ROI
    def on_down(self, event):
        if self.bgr_original is None:
            return
        cx, cy = self.disp_to_orig(event.x, event.y)
        self._center_orig = (cx, cy)
        self._dragging = True
        self._oval_id = None
    def on_drag(self, event):
        if not self._dragging or self.bgr_original is None or self._center_orig is None:
            return
        cx, cy = self._center_orig
        ox, oy = self.disp_to_orig(event.x, event.y)
        r = int(((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5)
        # Отображение на холсте
        dcx, dcy = self.orig_to_disp(cx, cy)
        dr = int(r * self.display_scale)
        # Удаление старого круга
        if self._oval_id is not None:
            self.canvas.delete(self._oval_id)
        # Отрисовка нового
        self._oval_id = self.canvas.create_oval(dcx - dr, dcy - dr, dcx + dr, dcy + dr,outline="yellow", width=3)
    def on_up(self, event):
        if not self._dragging or self.bgr_original is None or self._center_orig is None:
            return
        cx, cy = self._center_orig
        ox, oy = self.disp_to_orig(event.x, event.y)
        r = int(((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5)
        self._dragging = False
        self._center_orig = None
        # Если радиус слишком маленький - считаем, что ROI не выделен
        if r < 10:
            self.roi_circle_orig = None
        else:
            self.roi_circle_orig = (cx, cy, r)
        self.redraw()

# Главное окно

class EyeComparisonApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Помутнение роговицы — U-Net + круглый ROI")
        self.root.geometry("1750x960")
        # Режим: одиночный анализ или сравнение до/после
        self.single_mode = tk.BooleanVar(value=False)
        # Хранение загруженных изображений
        self.before_pil = None
        self.after_pil = None
        # Загрузкамодели
        self.unet_model, self.unet_device, msg = load_unet_model(MODEL_PATH)
        self.unet_status_msg = msg
        # Создание интерфейса
        self.create_widgets()
        self.update_mode()
        if self.unet_model is None:
            messagebox.showwarning("U-Net", self.unet_status_msg)
    def create_widgets(self):
        # Верхняя панель с кнопками
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
            top, text="Одиночный анализ", variable=self.single_mode, command=self.update_mode
        )
        self.mode_check.pack(side="right", padx=10)
        # Панель настроек
        params = ttk.LabelFrame(top, text="Настройки", padding=8)
        params.pack(side="right", padx=5)
        ttk.Button(params, text="Сброс ROI", command=self.reset_roi).grid(row=0, column=0, sticky="we", pady=(2, 0))
        ttk.Button(params, text="Показать исходник", command=self.reset_view).grid(row=0, column=1, sticky="we",pady=(2, 0))
        # Основная область с двумя панелями изображений
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.before_panel = ImagePanel(left, "До операции (большой снимок)")
        self.before_panel.pack(fill="both", expand=True, pady=5)
        self.after_panel = ImagePanel(left, "После операции (большой снимок)")
        self.after_panel.pack(fill="both", expand=True, pady=5)
        # Правая панель с результатами
        right = ttk.Frame(main)
        right.pack(side="right", fill="y")
        results_box = ttk.LabelFrame(right, text="Результаты анализа", padding=8)
        results_box.pack(fill="x")
        self.results_text = tk.Text(results_box, width=48, height=28, wrap="word")
        self.results_text.pack()
    def update_mode(self):
        if self.single_mode.get():
            # Одиночный режим
            self.after_panel.pack_forget()
            self.btn_after.config(state="disabled")
            self.btn_before.config(text="Снимок")
            self.btn_run.config(text="Анализировать")
            self.instruction_label.config(text="Загрузите снимок и выделите круг мышкой (клик — центр, протяжка — радиус)")
        else:
            # Режим сравнения
            if not self.after_panel.winfo_ismapped():
                self.after_panel.pack(fill="both", expand=True, pady=5)
            self.btn_after.config(state="normal")
            self.btn_before.config(text="До операции")
            self.btn_run.config(text="Сравнить")
            self.instruction_label.config(
                text="Загрузите ДО и ПОСЛЕ, выделите круг ROI (если надо) и нажмите «Сравнить»")
        # Очистить результаты при смене режима
        self.results_text.delete("1.0", "end")
    def reset_roi(self):
        self.before_panel.reset_roi()
        self.after_panel.reset_roi()
    def reset_view(self):
        self.before_panel.reset_view_to_original()
        self.after_panel.reset_view_to_original()
    def load_image(self, which):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        img = Image.open(path).convert("RGB")
        if which == "before":
            self.before_pil = img
            self.before_panel.set_image_pil(img)
        else:
            self.after_pil = img
            self.after_panel.set_image_pil(img)
    def run_analysis(self):
        try:
            if self.single_mode.get():
                self.run_single()
            else:
                self.run_compare()
        except Exception as e:
            import traceback
            messagebox.showerror("Ошибка при анализе", f"{e}\n\n{traceback.format_exc()}")
    def _analyze_one(self, bgr: np.ndarray, roi_mask255: np.ndarray):
        if self.unet_model is None:
            raise RuntimeError(self.unet_status_msg)
        mask255, thr, p95, min_area = unet_predict_mask_255(self.unet_model, self.unet_device, bgr, roi_mask255)
        percent, mask_roi = compute_percent_from_mask(mask255, roi_mask255)
        # Если процент меньше шумового порога - считать, что помутнения нет
        if percent < NOISE_FLOOR_PERCENT:
            percent = 0.0
            mask_roi[:] = 0
        info = f"U-Net (порог={thr:.2f}, уверенность={p95:.2f}, мин.площадь={min_area})"
        return percent, mask_roi, info
    def run_single(self):
        if self.before_pil is None:
            messagebox.showwarning("Ошибка", "Загрузите снимок")
            return
        bgr = pil_to_bgr(self.before_pil)
        roi_mask = self.before_panel.get_roi_mask()
        roi_circle = self.before_panel.roi_circle_orig
        p, mask_roi, method_info = self._analyze_one(bgr, roi_mask)
        # Показ результата с наложенной маской
        shown = overlay_mask_bgr(bgr, mask_roi, alpha=OVERLAY_ALPHA)
        shown = draw_roi_circle(shown, roi_circle)
        self.before_panel.show_bgr(shown)
        # Вывод результатов в текстовое поле
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"Метод: {method_info}\n")
        self.results_text.insert("end", f"Модель: {MODEL_PATH}\n")
        self.results_text.insert("end", f"ROI: {'выбран' if roi_circle else 'не выбран (анализ всей картинки)'}\n")
        self.results_text.insert("end", f"Шумовой порог: < {NOISE_FLOOR_PERCENT:.1f}% -> 0%\n\n")
        self.results_text.insert("end", f"ПОМУТНЕНИЕ: {p:.2f}%\n")
    def run_compare(self):
        if self.before_pil is None or self.after_pil is None:
            messagebox.showwarning("Ошибка", "Загрузите оба снимка")
            return
        b1 = pil_to_bgr(self.before_pil)
        b2 = pil_to_bgr(self.after_pil)
        roi_mask1 = self.before_panel.get_roi_mask()
        roi_circle1 = self.before_panel.roi_circle_orig
        roi_mask2 = self.after_panel.get_roi_mask()
        roi_circle2 = self.after_panel.roi_circle_orig
        p1, m1, info1 = self._analyze_one(b1, roi_mask1)
        p2, m2, info2 = self._analyze_one(b2, roi_mask2)
        diff = p1 - p2
        # Показываем результаты на панелях
        shown1 = overlay_mask_bgr(b1, m1, alpha=OVERLAY_ALPHA)
        shown1 = draw_roi_circle(shown1, roi_circle1)
        self.before_panel.show_bgr(shown1)
        shown2 = overlay_mask_bgr(b2, m2, alpha=OVERLAY_ALPHA)
        shown2 = draw_roi_circle(shown2, roi_circle2)
        self.after_panel.show_bgr(shown2)
        # Выводим сравнение
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"ДО: {info1}\n")
        self.results_text.insert("end", f"ПОСЛЕ: {info2}\n")
        self.results_text.insert("end", f"Модель: {MODEL_PATH}\n")
        self.results_text.insert("end", f"Шумовой порог: < {NOISE_FLOOR_PERCENT:.1f}% -> 0%\n\n")
        self.results_text.insert("end", f"ROI ДО: {'выбран' if roi_circle1 else 'вся картинка'}\n")
        self.results_text.insert("end", f"ROI ПОСЛЕ: {'выбран' if roi_circle2 else 'вся картинка'}\n\n")
        self.results_text.insert("end", f"ДО: {p1:.2f}%\n")
        self.results_text.insert("end", f"ПОСЛЕ: {p2:.2f}%\n")
        self.results_text.insert("end", f"Разница: {diff:.2f}%\n")


# Запуск

root = tk.Tk()
app = EyeComparisonApp(root)
root.mainloop()