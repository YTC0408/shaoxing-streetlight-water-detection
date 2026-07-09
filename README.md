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
| `datasets/damaged_lights` | 目标检测 | `Not Working` / `Working` | 1581 / 92 |
| `datasets/dataset_new` | 实例分割（多边形） | `water` | 1200 / 300 |

> 积水数据集原目录名为 `lables`，已修正为 `labels` 并补齐 `data.yaml`。

## 训练

```bash
pip install ultralytics opencv-python

python train.py lights   # 路灯故障（检测）
python train.py water    # 路面积水（分割，用 yolov8n-seg.pt）
python train.py          # 两个都训
```

训练好的权重在 `runs/` 下，把 `yolo-v.py` 中的 `YOLO('yolov8n.pt')` 改成对应权重即可推理。

> `.pt` 权重与 `runs/` 已在 `.gitignore` 中忽略。
