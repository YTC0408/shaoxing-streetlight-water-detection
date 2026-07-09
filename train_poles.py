"""路灯整杆数据集(夜间街景,含灯杆)训练脚本。
数据集: pole_dst2328
类别  : lightening(整根灯亮), damaged pole(杆体损坏)
注意  : train 中 damaged pole 仅 ~3%,类别严重不平衡。
        默认不开 class-weight,先跑 baseline 看看 mAP;
        后续若 damaped pole recall 很低,在 data.yaml 加:
          train: images/train   (重采样) 或 训练时传 cos_lr + close_mosaic。
        现在先求能跑通,不强求指标。
"""
from ultralytics import YOLO

def train_poles():
    model = YOLO('yolov8n.pt')
    model.train(
        data='datasets/pole_dst2328/data.yaml',
        epochs=100, imgsz=640, name='poles',
    )

if __name__ == '__main__':
    train_poles()
