"""
路灯故障检测 + 边缘截断判定
- bbox 触碰画面四边 → uncertain(橙框),不输出亮/灭
- bbox 完整 → 按类别 Working / Not Working 显示
运行: conda run -n yolo python detect_lights.py
按 q 退出
"""
import cv2
from ultralytics import YOLO

EDGE_MARGIN = 5  # ponytail: 像素常量,按分辨率调;灯头在画面上半部,底边触边几乎不发生
# 路灯权重(未训练时回退 yolov8n.pt,届时 Working/Not Working 类不存在,需自训)
WEIGHTS = 'runs/detect/lights/weights/best.pt'


def is_truncated(x1, y1, x2, y2, w, h, margin=EDGE_MARGIN):
    """任一边距画面边缘 ≤ margin 即视为拍不全。"""
    return (x1 <= margin or y1 <= margin
            or x2 >= w - margin or y2 >= h - margin)


def demo():
    """自检:边界框判 uncertain,居中框判正常。"""
    w, h = 640, 480
    assert is_truncated(0, 100, 50, 200, w, h) is True, '左触边应截断'
    assert is_truncated(100, 0, 200, 50, w, h) is True, '上触边应截断'
    assert is_truncated(200, 100, w, 200, w, h) is True, '右触边应截断'
    assert is_truncated(100, 200, 200, h, w, h) is True, '下触边应截断'
    assert is_truncated(200, 100, 400, 300, w, h) is False, '居中框应正常'
    print('is_truncated self-check OK')


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, label, (int(x1), max(int(y1) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    demo()
    try:
        model = YOLO(WEIGHTS)
    except FileNotFoundError:
        print(f'[!] 路灯权重 {WEIGHTS} 不存在,回退 yolov8n.pt(无路灯类别,仅作演示)')
        model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        results = model(frame, verbose=False)
        names = results[0].names

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if is_truncated(x1, y1, x2, y2, w, h):
                draw_box(frame, x1, y1, x2, y2, 'uncertain', (0, 165, 255))  # orange
            elif cls_name in ('light_on', 'Working'):
                draw_box(frame, x1, y1, x2, y2, 'light_on', (0, 255, 0))
            elif cls_name in ('light_off', 'Not Working', 'damaged pole'):
                draw_box(frame, x1, y1, x2, y2, 'light_off', (0, 0, 255))
            else:
                draw_box(frame, x1, y1, x2, y2, cls_name, (255, 255, 0))

        cv2.imshow('Lights Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
