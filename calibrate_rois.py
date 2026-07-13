"""ROI 标定工具:从摄像头抓一帧,鼠标拖框给每盏灯圈位置,保存到 rois.json。
手动模式:逐盏鼠标拖框。
自动模式(--auto):跑 OBB 推理一次,把所有非截断检测框作为候选,人用 selectROI 微调或剔除。
用法:
  python calibrate_rois.py --source 0 --output rois.json
  python calibrate_rois.py --auto --source 0 --output rois.json  # 候选预生成 + 人工微调
  python calibrate_rois.py --demo  # 合成一张测试图,自检保存/加载
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
EXAMPLE_ROIS = ROOT / "rois.example.json"
# 候选框半宽默认值(像素);人是后续微调
DEFAULT_HALF = 40


def load_existing(path):
    if not Path(path).exists():
        return {"frame_size": None, "weights": None, "lights": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_rois(path, frame_size, lights, weights=None):
    data = {"frame_size": list(frame_size), "lights": lights}
    if weights:
        data["weights"] = weights
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def grab_frame(source):
    """支持 int(摄像头)、字符串(rtsp/视频文件路径)。"""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("读帧失败")
    return frame


def auto_detect_lights(frame, weights, conf=0.25):
    """跑 OBB 推理一次,返回 ROI 候选列表(去重叠, 去边缘截断)。
    每项: (cx, cy, half_w, half_h, cls_id)"""
    from ultralytics import YOLO
    from detect_lights_wbf import predict_frame_obb, is_truncated
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[!] 加载权重 {weights} 失败: {e};改为 yolov8n-obb.pt", file=sys.stderr)
        model = YOLO("yolov8n-obb.pt")
    polys, scores, labels = predict_frame_obb([("auto", model)], frame, conf_thr=conf)
    h, w = frame.shape[:2]
    cands = []
    for pts, sc, cls in zip(polys, scores, labels):
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        if is_truncated(x1, y1, x2, y2, w, h):
            continue  # ponytail: 触边检测框不可靠,排除
        hw = max(DEFAULT_HALF, (x2 - x1) // 2)
        hh = max(DEFAULT_HALF, (y2 - y1) // 2)
        cands.append((int(cx), int(cy), int(hw), int(hh), int(cls)))
    # 简单 NMS:中心点距离 < 80 视为同一灯
    cands = _nms_candidates(cands, dist_thr=80)
    return cands


def _nms_candidates(cands, dist_thr=80):
    """按 (x+y) 字典序去重叠;距离阈值内只保留第一个。"""
    kept = []
    for c in cands:
        if all(abs(c[0] - k[0]) + abs(c[1] - k[1]) > dist_thr for k in kept):
            kept.append(c)
    return kept


def demo():
    """合成一张含 2 个色块的图,跑标定主流程(无 GUI):保存 rois.json 再读回校验。
    --auto 模式自检:用 _stub_predict 替身验证 NMS + 候选生成。
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (180, 220), (260, 280), (0, 255, 0), 2)  # 模拟 light_on
    cv2.rectangle(img, (560, 220), (640, 280), (0, 0, 255), 2)  # 模拟 light_damage 触边
    lights = [
        {"id": "L1", "name": "示例灯1", "cx": 220, "cy": 250, "half_w": 40, "half_h": 30},
        {"id": "L2", "name": "示例灯2(触边)", "cx": 600, "cy": 250, "half_w": 40, "half_h": 30},
    ]
    out = ROOT / "_rois_demo.json"
    save_rois(out, (640, 480), lights)
    # 回读
    loaded = load_existing(out)
    assert loaded["frame_size"] == [640, 480]
    assert len(loaded["lights"]) == 2
    assert loaded["lights"][0]["id"] == "L1"
    out.unlink()

    # NMS 单元测试
    cs = [(100, 100, 40, 40, 1), (110, 105, 40, 40, 1), (300, 200, 40, 40, 0)]
    kept = _nms_candidates(cs, dist_thr=80)
    assert len(kept) == 2, f"NMS 应保留 2 个,实际 {len(kept)}"
    assert kept[0] == (100, 100, 40, 40, 1)
    assert kept[1] == (300, 200, 40, 40, 0)
    print("calibrate_rois demo OK (含 NMS 自检)")


def _draw_candidates(frame, lights):
    """把所有候选灯画到 frame 上(标号 + 矩形),返回新 frame。"""
    out = frame.copy()
    for i, l in enumerate(lights):
        cx, cy = l["cx"], l["cy"]
        hw, hh = l.get("half_w", DEFAULT_HALF), l.get("half_h", DEFAULT_HALF)
        x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(out, l["id"], (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out


def _interactive_refine(frame, cands, h, w):
    """对每个候选,弹 selectROI 让用户拖框微调;按 d 删除当前,按 n 跳过。
    弹窗初始 ROI = 候选框;Enter 接受(可拖动),c 取消(保留原值),Esc 退出整个流程。
    """
    lights = []
    for i, (cx, cy, hw, hh, cls) in enumerate(cands):
        # 初始 ROI(给 selectROI 起点)
        x0 = max(0, cx - hw)
        y0 = max(0, cy - hh)
        w0 = min(hw * 2, w - x0)
        h0 = min(hh * 2, h - y0)
        # 准备预览
        preview = frame.copy()
        cv2.rectangle(preview, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 255), 2)
        cv2.putText(preview, f"[{i+1}/{len(cands)}] cls={cls} 拖框微调,Enter 接受,c 跳过,d 删除,Esc 退出",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("calibrate", preview)
        print(f"\n[{i+1}/{len(cands)}] 候选 cls={cls} 中心=({cx},{cy})")
        x, y, ww, hh2 = cv2.selectROI("calibrate", preview, fromCenter=False, showCrosshair=True)
        if ww == 0 or hh2 == 0:
            # selectROI 按 c 取消时返回 (0,0,0,0) — 视为跳过(保留候选)
            print(f"  [跳过] 保留原候选 ({cx},{cy})")
            lights.append({
                "id": f"L{len(lights) + 1}", "name": f"灯{len(lights) + 1}",
                "cx": cx, "cy": cy, "half_w": hw, "half_h": hh,
            })
            continue
        ncx, ncy = x + ww // 2, y + hh2 // 2
        lights.append({
            "id": f"L{len(lights) + 1}", "name": f"灯{len(lights) + 1}",
            "cx": int(ncx), "cy": int(ncy),
            "half_w": int(ww // 2), "half_h": int(hh2 // 2),
        })
        print(f"  [+] 接受: cx={ncx}, cy={ncy}, wh=({ww},{hh2})")
    return lights


def main():
    ap = argparse.ArgumentParser(description="路灯 ROI 标定")
    ap.add_argument("--source", type=str, default="0", help="摄像头索引/RTSP/视频/图片")
    ap.add_argument("--output", type=str, default="rois.json")
    ap.add_argument("--weights", type=str, default="runs/obb/runs/obb/kfold_fold0/weights/best.pt",
                    help="OBB 权重,auto 模式必用")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="跑 OBB 推理一次生成候选,人用 selectROI 微调/删除")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    print(f"[*] 从 {args.source} 抓首帧...")
    frame = grab_frame(args.source)
    h, w = frame.shape[:2]
    print(f"[*] 画面 {w}x{h}")

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    if args.auto:
        print(f"[*] --auto: 跑 OBB 推理生成候选(权重={args.weights}, conf={args.conf})...")
        cands = auto_detect_lights(frame, args.weights, conf=args.conf)
        print(f"[*] 检测到 {len(cands)} 个候选(去截断 + NMS)")
        if not cands:
            print("[!] 无候选,可手动模式(去掉 --auto)逐盏框选")
            return
        lights = _interactive_refine(frame, cands, h, w)
    else:
        # 手动模式(原逻辑)
        existing = load_existing(args.output)
        lights = list(existing.get("lights", []))
        cv2.imshow("calibrate", frame)
        print("\n操作说明:")
        print("  a 添加新灯(拖框)/ d 删最后一盏 / s 保存 / q 不保存退出 / c 重抓一帧\n")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("[*] 不保存,退出")
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                frame = grab_frame(args.source)
                cv2.imshow("calibrate", frame)
                print("[*] 重抓一帧")
            elif key == ord('a'):
                x, y, ww, hh = cv2.selectROI("calibrate", frame, fromCenter=False, showCrosshair=True)
                if ww == 0 or hh == 0:
                    continue
                cx, cy = x + ww // 2, y + hh // 2
                lights.append({
                    "id": f"L{len(lights) + 1}", "name": f"灯{len(lights) + 1}",
                    "cx": int(cx), "cy": int(cy),
                    "half_w": int(ww // 2), "half_h": int(hh // 2),
                })
                preview = _draw_candidates(frame, lights)
                cv2.imshow("calibrate", preview)
            elif key == ord('d'):
                if lights:
                    lights.pop()
                    preview = _draw_candidates(frame, lights)
                    cv2.imshow("calibrate", preview)
            elif key == ord('s'):
                break

    # 保存
    save_rois(args.output, (w, h), lights, args.weights)
    print(f"[*] 已保存到 {args.output},共 {len(lights)} 盏灯")
    cv2.destroyAllWindows()
