"""双权重 WBF 推理 UI 工具。
功能:
  - 单张图片推理(选文件 → 显示标注图 + 检测结果表格)
  - 批量推理(选目录 → 跑完输出 mAP/计数汇总)
  - 双权重加载: lights_only + poles, WBF 融合(可独立关闭)
  - 边缘截断判定 → uncertain 单独标记
  - 类别: light_off / light_on(2 类,模型内 remap)
依赖: ultralytics, opencv-python, Pillow, tkinter(内置)
运行: python test_ui.py
"""
import os, sys, glob, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

from ultralytics import YOLO
from detect_lights_wbf import (
    weighted_box_fusion, is_truncated, EDGE_MARGIN, WEIGHTS_LIST, CONF
)

# ----------- 核心:复用 detect_lights_wbf 的预测逻辑 -------------

REMAP = {'light_on': 1, 'Working': 1, 'lightening': 1,
         'light_off': 0, 'Not Working': 0, 'damaged pole': 0}
COLORS = {'light_on': (0, 200, 0), 'light_off': (0, 0, 220),
          'uncertain': (0, 165, 255), 'gt': (255, 128, 0)}


def predict_frame(models, frame, conf_thr=0.25, use_wbf=True):
    """单帧推理: 返回 (boxes, scores, labels) numpy + per-model 原始(可选)
    """
    boxes_all, scores_all, labels_all, weights_all = [], [], [], []
    per_model = []  # [(name, boxes, scores, labels), ...]
    for name, model in models:
        res = model(frame, conf=conf_thr, verbose=False)[0]
        b = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        s = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        l = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros(0, int)
        new_l = np.array([REMAP.get(res.names.get(c, ''), c) for c in l])
        per_model.append((name, b.copy(), s.copy(), new_l.copy()))
        boxes_all.append(b)
        scores_all.append(s)
        labels_all.append(new_l)
        weights_all.append(np.full(len(s), 1.0))

    if use_wbf and len(models) > 1:
        fb, fs, fl = weighted_box_fusion(boxes_all, scores_all, labels_all,
                                          iou_thr=0.55, weights=weights_all)
    else:
        # 单模型: 直接拼接
        if boxes_all:
            fb = np.concatenate(boxes_all)
            fs = np.concatenate(scores_all)
            fl = np.concatenate(labels_all)
        else:
            fb = np.zeros((0, 4))
            fs = np.zeros(0)
            fl = np.zeros(0, int)
    return fb, fs, fl, per_model


def annotate(frame, boxes, scores, labels):
    """画框 + uncertain 标记"""
    h, w = frame.shape[:2]
    out = frame.copy()
    for (x1, y1, x2, y2), sc, cls in zip(boxes, scores, labels):
        if is_truncated(x1, y1, x2, y2, w, h):
            draw_box(out, x1, y1, x2, y2, 'uncertain', COLORS['uncertain'])
        elif cls == 1:
            draw_box(out, x1, y1, x2, y2, f'light_on {sc:.2f}', COLORS['light_on'])
        elif cls == 0:
            draw_box(out, x1, y1, x2, y2, f'light_off {sc:.2f}', COLORS['light_off'])
    return out


def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img, label, (int(x1), max(int(y1) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --------------- UI 部分 ---------------

class TestApp:
    def __init__(self, root):
        self.root = root
        root.title('Lights WBF Test UI')
        root.geometry('1100x700')

        # 状态
        self.models = []  # [(name, YOLO)]
        self.use_wbf = tk.BooleanVar(value=True)
        self.conf_thr = tk.DoubleVar(value=0.25)
        self.last_image = None  # BGR np
        self.last_result = None  # (boxes, scores, labels, per_model)

        self._build_ui()
        self._load_models_async()

    def _build_ui(self):
        # 左侧:控件
        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text='模型加载', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        self.model_listbox = tk.Listbox(left, height=4, width=30)
        self.model_listbox.pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=6)

        ttk.Label(left, text='推理设置', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        ttk.Checkbutton(left, text='启用 WBF 融合', variable=self.use_wbf).pack(anchor='w')
        ttk.Label(left, text='置信度阈值:').pack(anchor='w', pady=(6, 0))
        ttk.Scale(left, from_=0.05, to=0.9, variable=self.conf_thr, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.conf_label = ttk.Label(left, text='0.25')
        self.conf_label.pack(anchor='w')
        self.conf_thr.trace_add('write', lambda *a: self.conf_label.config(
            text=f'{self.conf_thr.get():.2f}'))

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=6)

        ttk.Label(left, text='单张推理', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        ttk.Button(left, text='选择图片…', command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(left, text='重新推理当前图', command=self.rerun_image).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=6)

        ttk.Label(left, text='批量推理', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        ttk.Button(left, text='选择目录…', command=self.open_dir).pack(fill=tk.X, pady=2)
        self.batch_status = ttk.Label(left, text='空闲')
        self.batch_status.pack(anchor='w', pady=2)
        self.batch_progress = ttk.Progressbar(left, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(left, text='保存标注图', command=self.save_annot).pack(fill=tk.X, pady=2)

        # 右侧:图像 + 结果
        right = ttk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, bg='#222')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 底栏:结果
        bottom = ttk.Frame(self.root, padding=4)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_tree = ttk.Treeview(bottom, columns=('cls', 'conf', 'unc'),
                                         show='headings', height=5)
        self.result_tree.heading('cls', text='类别')
        self.result_tree.heading('conf', text='置信度')
        self.result_tree.heading('unc', text='是否 uncertain')
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.summary_label = ttk.Label(bottom, text='无结果')
        self.summary_label.pack(side=tk.LEFT, padx=8)

    def _load_models_async(self):
        def load():
            self.model_listbox.insert(tk.END, '加载中…')
            ok = []
            for name, path in WEIGHTS_LIST:
                try:
                    m = YOLO(path)
                    ok.append((name, m))
                    self.model_listbox.insert(tk.END, f'✓ {name}: {path}')
                except FileNotFoundError:
                    self.model_listbox.insert(tk.END, f'✗ {name}: 未找到 {path}')
            self.models = ok
            if not ok:
                messagebox.showwarning('模型缺失', '无可用模型,请确认 runs/detect/ 下的 best.pt')
        threading.Thread(target=load, daemon=True).start()

    def open_image(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All', '*.*')])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('错误', f'无法读取 {path}')
            return
        self.last_image = img
        self._run_predict(img, source_name=os.path.basename(path))

    def rerun_image(self):
        if self.last_image is None:
            messagebox.showinfo('提示', '请先选一张图')
            return
        self._run_predict(self.last_image, source_name='(已重新推理)')

    def _run_predict(self, img, source_name=''):
        if not self.models:
            messagebox.showwarning('模型未就绪', '请等待模型加载完成')
            return
        boxes, scores, labels, per_model = predict_frame(
            self.models, img, conf_thr=self.conf_thr.get(), use_wbf=self.use_wbf.get())
        self.last_result = (boxes, scores, labels, per_model)

        # 画图
        annot = annotate(img, boxes, scores, labels)
        self._show_image(annot)

        # 列表
        self.result_tree.delete(*self.result_tree.get_children())
        h, w = img.shape[:2]
        cnt_off = cnt_on = cnt_unc = 0
        for (x1, y1, x2, y2), sc, cls in zip(boxes, scores, labels):
            unc = is_truncated(x1, y1, x2, y2, w, h)
            cls_name = 'light_on' if cls == 1 else ('light_off' if cls == 0 else f'cls{cls}')
            self.result_tree.insert('', tk.END, values=(cls_name, f'{sc:.2f}', '是' if unc else '否'))
            if unc: cnt_unc += 1
            elif cls == 0: cnt_off += 1
            elif cls == 1: cnt_on += 1
        self.summary_label.config(
            text=f'{source_name}  light_off:{cnt_off}  light_on:{cnt_on}  uncertain:{cnt_unc}  '
                 f'WBF:{"开" if self.use_wbf.get() else "关"}  '
                 f'per_model:{[(n, len(b)) for n,b,_,_ in per_model]}')

    def _show_image(self, bgr):
        h, w = bgr.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = 800, 500
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(bgr, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img, anchor=tk.CENTER)

    def open_dir(self):
        d = filedialog.askdirectory(title='选择图片目录')
        if not d:
            return
        files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            files.extend(glob.glob(os.path.join(d, ext)))
        if not files:
            messagebox.showinfo('提示', '该目录无图片')
            return

        def run():
            self.batch_status.config(text=f'共 {len(files)} 张')
            self.batch_progress['maximum'] = len(files)
            cnt_off = cnt_on = cnt_unc = 0
            out_dir = os.path.join(d, '_annot')
            os.makedirs(out_dir, exist_ok=True)
            for i, f in enumerate(files):
                img = cv2.imread(f)
                if img is None:
                    continue
                boxes, scores, labels, _ = predict_frame(
                    self.models, img, conf_thr=self.conf_thr.get(), use_wbf=self.use_wbf.get())
                h, w = img.shape[:2]
                for (x1, y1, x2, y2), sc, cls in zip(boxes, scores, labels):
                    if is_truncated(x1, y1, x2, y2, w, h): cnt_unc += 1
                    elif cls == 0: cnt_off += 1
                    elif cls == 1: cnt_on += 1
                annot = annotate(img, boxes, scores, labels)
                cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), annot)
                self.batch_progress['value'] = i + 1
                self.batch_status.config(text=f'{i+1}/{len(files)}  off:{cnt_off} on:{cnt_on} unc:{cnt_unc}')
            self.batch_status.config(
                text=f'完成 {len(files)} 张 → {out_dir}  off:{cnt_off} on:{cnt_on} unc:{cnt_unc}')

        threading.Thread(target=run, daemon=True).start()

    def save_annot(self):
        if self.last_image is None or self.last_result is None:
            messagebox.showinfo('提示', '没有可保存的图')
            return
        path = filedialog.asksaveasfilename(defaultextension='.jpg',
                                             filetypes=[('JPG', '*.jpg')])
        if not path:
            return
        boxes, scores, labels, _ = self.last_result
        cv2.imwrite(path, annotate(self.last_image, boxes, scores, labels))
        messagebox.showinfo('已保存', path)


if __name__ == '__main__':
    root = tk.Tk()
    app = TestApp(root)
    root.mainloop()
