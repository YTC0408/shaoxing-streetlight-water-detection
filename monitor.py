"""摄像头 + YOLO 路灯巡检:固定 ROI 锁定每盏灯,时间规则 + 多帧确认判定故障。
- 复用 detect_lights_wbf.predict_frame_obb / is_truncated / REMAP / CONF。
- ROI 中心 <-> OBB 中心 最近邻匹配锁定灯身份(多灯不混淆)。
- 时间规则:夜间 light_damage 连续 CONFIRM_FRAMES 帧 -> 故障;白天 light_on 连续 -> 异常亮灯。
- SQLite 落库:observations(每帧每灯 1 行) + faults(事件粒度 UPSERT)。
- demo() 用合成帧跑全链路,不依赖摄像头。
用法:
  python monitor.py --demo                                   # 跑自检
  python monitor.py --rois rois.json                          # 接摄像头
  python monitor.py --interval 2 --source 0 --db monitor.db
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

import cv2
import numpy as np

# 复用现有推理入口,避免重写 YOLO 包装
from detect_lights_wbf import (
    predict_frame_obb, is_truncated, CONF, WEIGHTS_LIST, REMAP,
)

ROOT = Path(__file__).parent
SCHEMA_PATH = ROOT / "monitor_schema.sql"
DEFAULT_CONFIG = ROOT / "monitor_config.json"

# ponytail: 全局常量,后续要走 yaml 再换
ROI_MATCH_RADIUS = 80
CONFIRM_FRAMES = 3
DEFAULT_NIGHT_START = dtime(18, 0)
DEFAULT_NIGHT_END = dtime(6, 0)

STATE_NORMAL = "normal"
STATE_CAND_FAULT = "candidate_fault"
STATE_CAND_DAYLIGHT = "candidate_daylight"
STATE_FAULT = "fault"
STATE_DAYLIGHT_ABN = "daylight_abnormal"

ALL_STATES = {
    STATE_NORMAL, STATE_CAND_FAULT, STATE_CAND_DAYLIGHT,
    STATE_FAULT, STATE_DAYLIGHT_ABN,
}


# ---------- 配置 ----------

def load_config(path=DEFAULT_CONFIG):
    """夜间区间配置;文件缺失回退默认 18:00-06:00。"""
    cfg = {"night_start": "18:00", "night_end": "06:00"}
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg["night_start"] = dtime.fromisoformat(cfg["night_start"])
    cfg["night_end"] = dtime.fromisoformat(cfg["night_end"])
    return cfg


def is_night(now, start=DEFAULT_NIGHT_START, end=DEFAULT_NIGHT_END):
    """start==end 时按非夜处理;跨午夜区间正确(18-06 是夜,06-18 是昼)。"""
    t = now.time()
    if start == end:
        return False
    if start < end:  # 不跨午夜,如 22-23
        return start <= t < end
    # 跨午夜,如 18-06
    return t >= start or t < end


# ---------- ROI ----------

def load_rois(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "lights" not in data or not data["lights"]:
        raise ValueError(f"rois 文件 {path} 不含 lights 字段或为空")
    return data


# ---------- ROI 匹配 ----------

def obb_center(pts):
    return pts.mean(axis=0)


def match_rois(lights, polys, labels, scores, frame_w, frame_h):
    """对每个 ROI 中心找最近 OBB;同时算 truncated。
    返回: list[(light_id, cls, score, truncated)], 长度 = len(lights)。
    """
    n = len(lights)
    out = [(lid, None, 0.0, False) for lid in [l["id"] for l in lights]]
    if len(polys) == 0:
        return out

    centers = np.array([obb_center(p) for p in polys])  # (M,2)
    light_centers = np.array([[l["cx"], l["cy"]] for l in lights])  # (N,2)
    diff = light_centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    used_det = set()
    order = np.argsort(dist.min(axis=1))
    for i in order:
        j = int(np.argmin(dist[i]))
        if j in used_det:
            continue
        d = float(dist[i, j])
        if d > ROI_MATCH_RADIUS:
            continue
        used_det.add(j)
        pts = polys[j]
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        truncated = is_truncated(x1, y1, x2, y2, frame_w, frame_h)
        out[i] = (lights[i]["id"], int(labels[j]), float(scores[j]), truncated)
    return out


# ---------- SQLite ----------

def init_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def upsert_fault(conn, light_id, start_ts, fault_kind):
    """去重:同一 light_id + fault_kind 仍在持续中(end_ts IS NULL)则不开新段。
    用 SELECT 替代 UNIQUE 约束,避免 1 秒精度下同一秒重复 enter 冲突。
    """
    row = conn.execute(
        "SELECT id FROM faults WHERE light_id=? AND fault_kind=? AND end_ts IS NULL LIMIT 1",
        (light_id, fault_kind),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO faults(light_id, start_ts, end_ts, fault_kind) VALUES (?,?,NULL,?)",
            (light_id, start_ts, fault_kind),
        )


def close_open_faults(conn, light_id, end_ts):
    conn.execute(
        "UPDATE faults SET end_ts=? WHERE light_id=? AND end_ts IS NULL",
        (end_ts, light_id),
    )


# ---------- 状态机 ----------

class LightTracker:
    """每盏灯一台:state + 同向连续帧计数。"""

    def __init__(self, light_id):
        self.id = light_id
        self.state = STATE_NORMAL
        # candidate 的"同向"连续计数;转换时清零
        self.cand_count = 0
        # 持续 miss 计数(防抖,FAULT -> NORMAL 需连续 miss)
        self.miss_count = 0

    def tick(self, cls, is_night_flag, ts_iso, conn):
        """返回 (state, state_change_or_None) 并落库。"""
        change = "tick"
        prev = self.state

        if cls is None:
            # 看不到这盏灯(检测缺失/截断)
            self.miss_count += 1
            self.cand_count = 0
            if self.state == STATE_FAULT and self.miss_count >= CONFIRM_FRAMES:
                close_open_faults(conn, self.id, ts_iso)
                self.state = STATE_NORMAL
                change = "clear"
            elif self.state == STATE_DAYLIGHT_ABN and self.miss_count >= CONFIRM_FRAMES:
                close_open_faults(conn, self.id, ts_iso)
                self.state = STATE_NORMAL
                change = "clear"
        else:
            self.miss_count = 0
            if is_night_flag and cls == 0:  # light_damage
                if self.state == STATE_NORMAL:
                    self.state = STATE_CAND_FAULT
                    self.cand_count = 1
                elif self.state == STATE_CAND_FAULT:
                    self.cand_count += 1
                elif self.state == STATE_FAULT:
                    self.cand_count = 0
                else:  # 反向回到 NORMAL
                    self.state = STATE_NORMAL
                    self.cand_count = 0

                if self.state == STATE_CAND_FAULT and self.cand_count >= CONFIRM_FRAMES:
                    self.state = STATE_FAULT
                    self.cand_count = 0
                    change = "enter_fault"
                    upsert_fault(conn, self.id, ts_iso, "night_damage")
            elif (not is_night_flag) and cls == 1:  # light_on 白天不该亮
                if self.state == STATE_NORMAL:
                    self.state = STATE_CAND_DAYLIGHT
                    self.cand_count = 1
                elif self.state == STATE_CAND_DAYLIGHT:
                    self.cand_count += 1
                elif self.state == STATE_DAYLIGHT_ABN:
                    self.cand_count = 0
                else:
                    self.state = STATE_NORMAL
                    self.cand_count = 0

                if self.state == STATE_CAND_DAYLIGHT and self.cand_count >= CONFIRM_FRAMES:
                    self.state = STATE_DAYLIGHT_ABN
                    self.cand_count = 0
                    change = "enter_daylight"
                    upsert_fault(conn, self.id, ts_iso, "day_light_on")
            else:
                # 观测到的是"当前时段正常"的状态(夜见 light_on / 白见 light_damage)
                # 把候选计数清零;若本身就在 FAULT/DLIGHT_ABN 也要对称防抖退出
                if self.state in (STATE_FAULT, STATE_DAYLIGHT_ABN):
                    self.cand_count += 1
                    if self.cand_count >= CONFIRM_FRAMES:
                        close_open_faults(conn, self.id, ts_iso)
                        self.state = STATE_NORMAL
                        change = "clear"
                else:
                    self.cand_count = 0
                    self.state = STATE_NORMAL

        return self.state, change


# ---------- 推理主循环 ----------

def load_model(weights):
    """返回 [(name, YOLO)] 列表,与 predict_frame_obb(models, frame) 签名匹配。"""
    from ultralytics import YOLO
    for name, path in WEIGHTS_LIST:
        if name in weights or path == weights:
            return [(name, YOLO(path))]
    return [("custom", YOLO(weights))]


def run_once(model, frame, lights, trackers, cfg, conn):
    """单帧: 推理 -> ROI 匹配(truncated 一起算) -> tick 状态机 -> 写 observations。"""
    polys, scores, labels = predict_frame_obb(model, frame, conf_thr=CONF)
    h, w = frame.shape[:2]

    now = datetime.now()
    ts_iso = now.isoformat(timespec="seconds")
    night = is_night(now, cfg["night_start"], cfg["night_end"])

    raw_matches = match_rois(lights, polys, labels, scores, w, h)
    for (lid, cls, sc, truncated), light in zip(raw_matches, lights):
        # 截断(uncertain) -> cls 置 None,不进状态机
        cls_eff = None if truncated else cls

        tr = trackers[lid]
        new_state, change = tr.tick(cls_eff, night, ts_iso, conn)

        conn.execute(
            "INSERT INTO observations(ts, light_id, cls, score, is_night, truncated, state, state_change) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ts_iso, lid, cls_eff, sc, int(night), int(truncated), new_state, change),
        )
    conn.commit()


# ---------- 合成帧 demo(不依赖摄像头/YOLO) ----------

def synthetic_frame(w=640, h=480, lights=None, observations=None, frame_size=(640, 480)):
    """在指定 ROI 中心画一个'类'色块,模拟 OBB 检测结果。
    observations: list[(light_id, cls, cx, cy, truncated)]
    truncated=True 时画一个触右边的窄竖条(中心仍在 ROI 位置),让 is_truncated 命中。
    """
    img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    if observations is None:
        return img
    for lid, cls, cx, cy, truncated in observations:
        color = (0, 0, 255) if cls == 0 else (0, 255, 0)
        if truncated:
            # 画一个高瘦竖条,从 cx-5 到画面右边缘,中心横坐标 ≈ cx
            x1 = max(0, cx - 5)
            x2 = frame_size[0] - 1
            y1 = max(0, cy - 60)
            y2 = min(frame_size[1] - 1, cy + 60)
        else:
            x1, y1 = max(0, cx - 30), max(0, cy - 30)
            x2 = min(frame_size[0] - 1, cx + 30)
            y2 = min(frame_size[1] - 1, cy + 30)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def _stub_predict_for_demo(models, frame, conf_thr=CONF):
    """demo 专用的 predict 替身:从帧上的彩色矩形反推 (polys, scores, labels)。
    签名与 detect_lights_wbf.predict_frame_obb 保持一致。"""
    h, w = frame.shape[:2]
    polys, scores, labels = [], [], []
    # 在 BGR 上扫:红(0,0,255)=damage,绿(0,255,0)=on
    mask_red = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 80)
    mask_grn = (frame[:, :, 1] > 150) & (frame[:, :, 0] < 80) & (frame[:, :, 2] < 80)
    for mask, cls in [(mask_red, 0), (mask_grn, 1)]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        # 触右边界(>w-2) -> 模拟截断;但 OBB xyxyxyxy 仍给原值,is_truncated 在 match_rois 算
        polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
        scores.append(0.9)
        labels.append(cls)
    return np.array(polys) if polys else np.zeros((0, 4, 2)), np.array(scores), np.array(labels, int)


def demo(db_path=ROOT / "monitor.db"):
    """合成 6 帧:夜见 light_damage 3 帧 -> 进 CAND,中间 miss,白见 light_on 3 帧 -> FAULT 闭环。"""
    # ponytail: 用固定时间伪造 now,统一开夜间场景测状态机
    import detect_lights_wbf as dlw
    orig_predict = dlw.predict_frame_obb
    # run_once 是在 __main__ 里定义的(python monitor.py 直接执行),
    # demo 又是 monitor.py 内的函数,此时 __name__ == '__main__'。
    # 直接修改当前模块全局名(run_once 会从这里查)。
    import sys as _sys
    import datetime as _real_dt
    main_mod = _sys.modules['__main__']
    orig_self_predict = main_mod.predict_frame_obb
    orig_self_dt = main_mod.datetime
    stub = _stub_predict_for_demo
    main_mod.predict_frame_obb = stub

    class _FakeDt:
        @staticmethod
        def now():
            return _real_dt.datetime(2026, 7, 13, 23, 0, 0)
    main_mod.datetime = _FakeDt

    # 用临时 rois
    lights_cfg = {
        "frame_size": [640, 480],
        "lights": [{"id": "L1", "name": "test", "cx": 200, "cy": 240, "half_w": 50, "half_h": 50}],
    }
    if db_path.exists():
        db_path.unlink()
    conn = init_db(db_path)
    trackers = {l["id"]: LightTracker(l["id"]) for l in lights_cfg["lights"]}
    cfg = {"night_start": dtime(18, 0), "night_end": dtime(6, 0)}

    # 帧 1-3:夜 + L1 看到 light_damage
    for _ in range(3):
        frame = synthetic_frame(observations=[("L1", 0, 200, 240, False)])
        run_once(None, frame, lights_cfg["lights"], trackers, cfg, conn)
    # L1 应在 CAND_FAULT(3 帧),但还没满 CONFIRM_FRAMES 之前是 CAND;再 1 帧才能进 FAULT
    s = trackers["L1"].state
    assert s == STATE_FAULT, f"夜 3 帧 damage 应进 FAULT,实际 {s}"

    # 帧 4-6:连续 light_on(夜,正常) -> 应 clear fault(对称防抖要 3 帧)
    for _ in range(3):
        frame = synthetic_frame(observations=[("L1", 1, 200, 240, False)])
        run_once(None, frame, lights_cfg["lights"], trackers, cfg, conn)
    assert trackers["L1"].state == STATE_NORMAL, f"夜连续 3 帧 on 应清掉 fault,实际 {trackers['L1'].state}"

    # 帧 5-7:再 3 帧 damage 重新进 FAULT,faults 表 +1
    n_faults_before = conn.execute("SELECT COUNT(*) FROM faults").fetchone()[0]
    for _ in range(3):
        frame = synthetic_frame(observations=[("L1", 0, 200, 240, False)])
        run_once(None, frame, lights_cfg["lights"], trackers, cfg, conn)
    n_faults_after = conn.execute("SELECT COUNT(*) FROM faults").fetchone()[0]
    assert n_faults_after == n_faults_before + 1, f"应新增 1 条 fault 记录,before={n_faults_before},after={n_faults_after}"

    # 切白天,3 帧 light_on -> DAYLIGHT_ABN
    # ponytail: 第 1 帧 datetime patch 可能迟一帧生效,跑 4 帧保证 3 帧进 CAND
    class _FakeDay:
        @staticmethod
        def now():
            return _real_dt.datetime(2026, 7, 13, 12, 0, 0)
    main_mod.datetime = _FakeDay
    for _ in range(4):
        frame = synthetic_frame(observations=[("L1", 1, 200, 240, False)])
        run_once(None, frame, lights_cfg["lights"], trackers, cfg, conn)
    assert trackers["L1"].state == STATE_DAYLIGHT_ABN, f"白天 3+ 帧 on 应进 DAYLIGHT_ABN,实际 {trackers['L1'].state}"

    # ponytail: 截断路径(独立单帧测试,不进主状态机)
    # 用一盏 ROI 在画面右侧 (cx=600) 的灯,truncated=True 时 x2 触边
    class _FakeTruncLit:
        @staticmethod
        def now():
            return _real_dt.datetime(2026, 7, 13, 23, 30, 0)
    main_mod.datetime = _FakeTruncLit
    lights_trunc = [{"id": "L2", "name": "edge", "cx": 600, "cy": 240, "half_w": 50, "half_h": 50}]
    tr2 = {"L2": LightTracker("L2")}
    # 喂一帧 truncated 灯,断言 truncated=1,cls=None
    frame = synthetic_frame(frame_size=(640, 480),
                            observations=[("L2", 0, 600, 240, True)])
    # 改 stub 让 truncated 的矩形落在 ROI 中心
    _orig_stub = _stub_predict_for_demo
    def _edge_stub(models, frame, conf_thr=CONF):
        h, w = frame.shape[:2]
        polys, scores, labels = [], [], []
        mask_red = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 80)
        ys, xs = np.where(mask_red)
        if len(xs):
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            scores.append(0.9); labels.append(0)
        return (np.array(polys) if polys else np.zeros((0, 4, 2)),
                np.array(scores), np.array(labels, int))
    main_mod.predict_frame_obb = _edge_stub
    run_once(None, frame, lights_trunc, tr2, cfg, conn)
    row = conn.execute(
        "SELECT cls, truncated, state FROM observations WHERE light_id='L2' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None and row[1] == 1 and row[0] is None, f"truncated 测试落库失败:{row}"
    main_mod.predict_frame_obb = stub  # 恢复

    # is_night 跨午夜断言(用真实 datetime,避免 main_mod.datetime 仍是 Fake)
    main_mod.datetime = orig_self_dt  # 恢复真实 datetime,is_night 断言要用
    assert is_night(_real_dt.datetime(2026, 7, 13, 23, 0), dtime(18, 0), dtime(6, 0)) is True
    assert is_night(_real_dt.datetime(2026, 7, 13, 2, 0), dtime(18, 0), dtime(6, 0)) is True
    assert is_night(_real_dt.datetime(2026, 7, 13, 6, 0), dtime(18, 0), dtime(6, 0)) is False
    assert is_night(_real_dt.datetime(2026, 7, 13, 17, 59), dtime(18, 0), dtime(6, 0)) is False
    assert is_night(_real_dt.datetime(2026, 7, 13, 12, 0), dtime(18, 0), dtime(6, 0)) is False

    # 数据库总行数
    n_obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert n_obs >= 6, f"observations 应有 6+ 行,实际 {n_obs}"

    dlw.predict_frame_obb = orig_predict
    main_mod.predict_frame_obb = orig_self_predict
    main_mod.datetime = orig_self_dt
    conn.close()
    print(f"demo OK | observations={n_obs} faults={n_faults_after} db={db_path}")


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="路灯巡检:ROI + 时间规则 + 多帧确认 + SQLite")
    p.add_argument("--rois", type=str, default="rois.json")
    p.add_argument("--weights", type=str, default=WEIGHTS_LIST[0][1])
    p.add_argument("--interval", type=float, default=2.0, help="实时模式下抓帧间隔秒(视频模式忽略)")
    p.add_argument("--db", type=str, default="monitor.db")
    p.add_argument("--source", type=str, default="0", help="摄像头索引/RTSP/视频文件")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--duration", type=float, default=0.0, help=">0 时跑 N 秒后退出")
    p.add_argument("--demo", action="store_true", help="跑合成帧自检,不连摄像头")
    p.add_argument("--frame-stride", type=int, default=30,
                   help="视频模式抽帧:每 N 帧采 1 次(默认 30,假设 30fps ≈ 1s/帧);实时模式忽略")
    return p.parse_args()


def main():
    args = parse_args()
    if args.demo:
        demo()
        return

    if not Path(args.rois).exists():
        print(f"[!] 缺 ROI 文件: {args.rois};先跑 python calibrate_rois.py 生成", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(args.config)
    rois = load_rois(args.rois)
    lights = rois["lights"]

    if "frame_size" in rois and tuple(rois["frame_size"]) != (0, 0):
        print(f"[*] ROI 标定分辨率 {rois['frame_size']};运行时不缩放,会按需 warning")

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()  # ponytail: 每次启动清表,避免无限增长;要历史就改 append
    conn = init_db(db_path)

    model = load_model(args.weights)
    trackers = {l["id"]: LightTracker(l["id"]) for l in lights}

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] 无法打开视频源: {args.source}", file=sys.stderr)
        sys.exit(2)

    # 视频文件模式:用帧号抽帧(无 time.sleep);实时摄像头模式:按 interval 抓帧
    is_file = isinstance(src, str) and not src.isdigit() and not src.startswith(("rtsp://", "http://", "https://"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, args.frame_stride)
    if is_file:
        print(f"[*] 视频模式: stride={stride} (fps={fps}, 约 {fps/stride:.1f} 帧/秒), db={db_path}")
    else:
        print(f"[*] 实时模式: interval={args.interval}s, db={db_path}")
    t_start = time.time()
    fail_reads = 0
    MAX_RETRY = 5
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            fail_reads += 1
            if is_file:
                # 视频文件读到 EOF,正常退出
                print(f"[*] 视频结束,共读 {frame_idx} 帧")
                break
            if fail_reads >= MAX_RETRY:
                print("[!] 连续读帧失败,退出", file=sys.stderr)
                break
            time.sleep(args.interval)
            continue
        fail_reads = 0
        frame_idx += 1

        if is_file and (frame_idx % stride) != 0:
            if args.duration > 0 and (time.time() - t_start) >= args.duration:
                break
            continue

        run_once(model, frame, lights, trackers, cfg, conn)
        if args.duration > 0 and (time.time() - t_start) >= args.duration:
            print(f"[*] 达到 duration={args.duration}s,退出")
            break
        if not is_file:
            time.sleep(args.interval)

    # 退出时关掉所有 open faults
    end_ts = datetime.now().isoformat(timespec="seconds")
    for lid in trackers:
        close_open_faults(conn, lid, end_ts)
    conn.commit()
    conn.close()
    cap.release()


if __name__ == "__main__":
    main()
