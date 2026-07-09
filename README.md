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

## 模型

当前使用预训练的 `yolov8n.pt`（COCO 通用目标）。检测路灯故障 / 路面积水需自行采集校园数据集并训练：

```bash
yolo train data=streetlight_water.yaml model=yolov8n.pt epochs=100 imgsz=640
```

训练好的权重放到项目目录，把 `yolo-v.py` 中的 `YOLO('yolov8n.pt')` 改成对应权重。

> `.pt` 权重文件已在 `.gitignore` 中忽略，不纳入版本管理。
