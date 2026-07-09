"""WBF(Weighted Box Fusion)合并两个模型的预测。
两个权重都跑同一帧,合并输出同一目标的高质量 bbox。
推理代价: 1.5~2x 单模型(两个权重都加载 + 两次前向 + 融合)。
"""
import cv2
import numpy as np
from ultralytics import YOLO

EDGE_MARGIN = 5  # ponytail: 像素常量,按分辨率调
WEIGHTS_LIST = [
    ('lights_only', 'runs/detect/lights_only/weights/best.pt'),
    ('poles',       'runs/detect/poles/weights/best.pt'),
]
CONF = 0.25  # ponytail: WBF 配对阶段用低阈值,融合后再用 conf 过滤


def is_truncated(x1, y1, x2, y2, w, h, margin=EDGE_MARGIN):
    return (x1 <= margin or y1 <= margin
            or x2 >= w - margin or y2 >= h - margin)


def weighted_box_fusion(boxes_list, scores_list, labels_list,
                        iou_thr=0.55, skip_box_thr=0.0, weights=None):
    """简化版 WBF,实现核心融合逻辑。
    boxes_list: list of (N_i, 4) xyxy,每项一个模型的预测
    scores_list: list of (N_i,) 置信度
    labels_list: list of (N_i,) 类别 ID
    返回: (M,4) boxes, (M,) scores, (M,) labels
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    all_boxes, all_scores, all_labels, all_weights = [], [], [], []
    for boxes, scores, labels, w in zip(boxes_list, scores_list, labels_list, weights):
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_weights.append(np.full(len(scores), w))

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    w_arr = np.concatenate(all_weights)

    if skip_box_thr > 0:
        keep = scores >= skip_box_thr
        boxes, scores, labels, w_arr = boxes[keep], scores[keep], labels[keep], w_arr[keep]

    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # 按类别分组,每类单独融合
    fused_boxes, fused_scores, fused_labels = [], [], []
    for cls in np.unique(labels):
        mask = labels == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        cls_w = w_arr[mask]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_w = cls_w[order]

        clusters = []  # list of list of indices into cls_*
        for i, b in enumerate(cls_boxes):
            best_iou, best_j = 0.0, -1
            # 取每个 cluster 的首个框作为 ref
            for j, cluster in enumerate(clusters):
                ref_box = cls_boxes[cluster[0]]
                iou = _iou(b, ref_box)
                if iou > iou_thr and iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                clusters[best_j].append(i)
            else:
                clusters.append([i])

        # 对每个 cluster 加权融合
        for cluster in clusters:
            c_boxes = cls_boxes[cluster]
            c_scores = cls_scores[cluster]
            c_w = cls_w[cluster]
            weighted = (c_boxes * c_w[:, None]).sum(0) / c_w.sum()
            fused_score = c_scores.mean()
            fused_boxes.append(weighted)
            fused_scores.append(fused_score)
            fused_labels.append(cls)

    return (np.array(fused_boxes).reshape(-1, 4),
            np.array(fused_scores),
            np.array(fused_labels, dtype=int))


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, label, (int(x1), max(int(y1) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    # 加载两个模型
    models = []
    for name, path in WEIGHTS_LIST:
        try:
            models.append((name, YOLO(path)))
        except FileNotFoundError:
            print(f'[!] 跳过 {name}: {path} 不存在')
    if not models:
        print('无可用模型,回退 yolov8n.pt')
        models = [('coco', YOLO('yolov8n.pt'))]

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        boxes_all, scores_all, labels_all, weights_all = [], [], [], []
        for i, (name, model) in enumerate(models):
            res = model(frame, conf=CONF, verbose=False)[0]
            if len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy()
            s = res.boxes.conf.cpu().numpy()
            l = res.boxes.cls.cpu().numpy().astype(int)
            # 类别名映射:统一到 light_on/light_off
            remap = {'light_on': 1, 'Working': 1, 'lightening': 1,
                     'light_off': 0, 'Not Working': 0, 'damaged pole': 0}
            new_l = np.array([remap.get(res.names.get(c, ''), c) for c in l])
            boxes_all.append(b)
            scores_all.append(s)
            labels_all.append(new_l)
            weights_all.append(np.full(len(s), 1.0))

        boxes, scores, labels = weighted_box_fusion(
            boxes_all, scores_all, labels_all, iou_thr=0.55, weights=weights_all)

        for x1, y1, x2, y2, sc, cls in zip(boxes, scores, labels):
            if is_truncated(x1, y1, x2, y2, w, h):
                draw_box(frame, x1, y1, x2, y2, 'uncertain', (0, 165, 255))
            elif cls == 1:
                draw_box(frame, x1, y1, x2, y2, f'light_on {sc:.2f}', (0, 255, 0))
            elif cls == 0:
                draw_box(frame, x1, y1, x2, y2, f'light_off {sc:.2f}', (0, 0, 255))
            else:
                draw_box(frame, x1, y1, x2, y2, f'cls{cls} {sc:.2f}', (255, 255, 0))

        cv2.putText(frame, f'models: {[n for n,_ in models]}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Lights WBF', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
