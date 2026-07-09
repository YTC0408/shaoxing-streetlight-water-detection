# 绍兴大学路灯故障与路面积水视觉检测

基于 YOLOv8 的视觉检测项目，面向绍兴大学校园场景，检测**路灯故障**与**路面积水**两类目标。

## 功能

- 实时摄像头 / 视频推理（`yolo-v.py`）
- 使用 Ultralytics YOLOv8 模型，可替换为自训练权重

## 环境

```bash
pip install ultralytics opencv-python
```

## 运行

```bash
python yolo-v.py
```

默认调用摄像头（`cv2.VideoCapture(0)`）。按 `q` 退出。改用视频文件时把 `VideoCapture(0)` 换成视频路径。

## 数据集

放在 `datasets/`（图片/标签体积大，未入库，仅保留 `data.yaml`）：

| 数据集 | 任务 | 类别 | 训练/验证 |
|---|---|---|---|
| `datasets/damaged_lights` | 目标检测 | `light_off` / `light_on` (原 Not Working/Working) | 1581 / 92 |
| `datasets/dataset_new` | 实例分割（多边形） | `water` | 1200 / 300 |
| `datasets/pole_dst2328` | 目标检测（含灯杆,夜间街景） | `lightening` / `damaged pole` (语义=light_on/light_off) | 5235 / 272 / 42 |

> 夜间场景下,灯不亮 = 故障。`damaged_lights` 与 `pole_dst2328` 类别语义对齐:`light_off` ≡ `damaged pole`,`light_on` ≡ `lightening`。
> 训练时各自独立用 `data.yaml`;`detect_lights.py` 的类别映射已支持两套命名。

> 积水数据集原目录名为 `lables`，已修正为 `labels` 并补齐 `data.yaml`。
> `pole_dst2328` 类别严重不平衡：train 中 `damaged pole` 仅 ~3%，先跑 baseline 再决定是否加权/重采样。

## 训练

```bash
pip install ultralytics opencv-python

python train.py lights      # 路灯故障（检测,只框灯头）
python train.py water       # 路面积水（分割,用 yolov8n-seg.pt）
python train_poles.py       # 路灯整杆（夜间街景,含灯杆,类别不平衡）
```

## 推理

- 单一模型:`python detect_lights.py`(默认 `WEIGHTS = 'runs/detect/lights/weights/best.pt'`,改路径切模型)
- 双模型 WBF 融合:`python detect_lights_wbf.py`,同时跑 `lights_only` + `poles` 两套权重,WBF 合并预测召回↑精度↑

按 `q` 退出。
