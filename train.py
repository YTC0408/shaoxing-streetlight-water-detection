from ultralytics import YOLO

# 路灯故障检测（目标检测，2 类：Not Working / Working）
def train_lights():
    model = YOLO('yolov8n.pt')
    model.train(data='datasets/damaged_lights/data.yaml',
                epochs=100, imgsz=640, name='lights')

# 路面积水检测（实例分割，多边形标注，1 类：water）
def train_water():
    model = YOLO('yolov8n-seg.pt')
    model.train(data='datasets/dataset_new/data.yaml',
                epochs=100, imgsz=640, name='water')

if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('lights', 'both'):
        train_lights()
    if which in ('water', 'both'):
        train_water()
