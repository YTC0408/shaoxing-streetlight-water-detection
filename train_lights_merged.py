"""合并路灯数据集训练。
6816 train / 364 val。light_on 多于 light_off(亮灯样本远多于故障),
训练后重点看 light_off 的 mAP/recall。"""
from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')
    model.train(
        data='datasets/lights_merged/data.yaml',
        epochs=100, imgsz=640, name='lights_merged',
    )

if __name__ == '__main__':
    train()
