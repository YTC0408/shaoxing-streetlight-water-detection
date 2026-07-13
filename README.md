# 绍兴大学路灯故障与路面积水视觉检测

基于 YOLOv8 的视觉检测项目，面向绍兴大学校园场景，检测**路灯故障**（OBB 定向框）与**路面积水**（实例分割）两类目标。

## 功能

- 路灯故障实时摄像头 / 视频推理（OBB 旋转框）
- 路面积水实时推理（`yolo-v.py`）
- Tkinter 测试工具 `test_ui.py`：单张 / 批量推理、置信度滑块、标注图 + 结果表
- 3-fold 交叉验证评估整杆歪杆识别效果（`kfold_train.py`）
- 路灯巡检 `monitor.py`：固定 ROI 锁定每盏灯、时间规则 + 多帧确认判定故障、SQLite 落库
- ROI 标定 `calibrate_rois.py`：手动逐盏框选 / `--auto` 半自动（OBB 推理预生成候选 + 人工微调）

## 环境

```bash
pip install ultralytics opencv-python Pillow   # tkinter 为 Python 内置
```

路灯训练/推理必须在 yolo conda 环境（已装 ultralytics + CUDA）：

```bash
D:\app\Python\anaconda\envs\yolo\python.exe monitor.py --demo
```

## 数据集

放在 `datasets/`（图片/标签体积大，未入库，仅保留 `data.yaml`）：

| 数据集 | 任务 | 类别 | 训练/验证 |
|---|---|---|---|
| `datasets/damaged_lights` | 目标检测 (HBB) | `light_off` / `light_on` (原 Not Working/Working) | 1581 / 92 |
| `datasets/dataset_new` | 实例分割（多边形） | `water` | 1200 / 300 |
| `datasets/pole_dst2328` | 目标检测 (HBB,夜间街景) | `lightening` / `damaged pole` | 5235 / 272 / 42 |
| `datasets/new_light` | 定向框 (OBB, 歪杆) | `light_damage` / `light_on` (Label Studio 导出) | 10 / 0 |
| `datasets/lights_merged` | 定向框 (OBB) | `light_damage` / `light_on` | 6860 / 366 |

> 类别语义统一：`light_damage` ≡ 灯灭/故障/歪杆，`light_on` ≡ 灯亮/正常。夜间场景下灯不亮即故障。
> `pole_dst2328` 中 `damaged pole` 原语义与 `light_off` 一致；`merge_lights.py` 在合并时做 ID 翻转 remap。
> `merge_lights.py` 把 HBB 数据集转轴对齐四边形(8 坐标)与 `new_light` 真实斜框一起并入 OBB 训练集。

## 训练

```bash
# 路灯故障: 合并三数据集, OBB 训练(单权重)
python merge_lights.py              # 重建 datasets/lights_merged(若已存在可跳过)
python augment_new_light.py         # 对 new_light 整杆样本做 OBB 离线增强(36 变体)
python train_lights_merged.py       # 75 轮, 早停+正则, 输出 runs/obb/lights_merged_obb/

# 路灯故障: 3-fold 交叉验证(整杆 recall 是关键指标)
python kfold_prep.py                # 生成 datasets/kfold/fold{k}.yaml
python kfold_train.py               # 3 折 × 75 轮 ≈ 4.5h

# 路面积水
python train.py                     # yolov8n-seg.pt, 100 轮
```

> 路灯训练必须使用 `D:\app\Python\anaconda\envs\yolo` conda 环境(已装 ultralytics + CUDA)。

## 推理

```bash
python detect_lights_wbf.py         # OBB 单模型摄像头推理, 画旋转四边形
python test_ui.py                   # Tkinter GUI: 单/批量, 置信度滑块
python yolo-v.py                    # 积水推理(默认 yolov8n.pt)
```

按 `q` 退出。

## 巡检（路灯故障定时检测 + 状态记录）

适用场景：摄像头/录像固定监控一段路的多盏路灯，定时抓帧分析，自动记录每盏灯的状态变化与故障事件。

```bash
# 1. 标定 ROI(只做一次)
python calibrate_rois.py --source 0 --output rois.json            # 手动:逐盏鼠标框选
python calibrate_rois.py --auto --source 0 --output rois.json     # 半自动:OBB 推理预生成候选 + 人工微调

# 2. 配置夜间区间(可选,默认 18:00-06:00;文件不存在时回退默认)
# 写入 monitor_config.json:
#   {"night_start": "18:00", "night_end": "06:00"}

# 3. 启动巡检
python monitor.py --rois rois.json                                # 实时摄像头, 间隔 2s
python monitor.py --source <video.mp4> --frame-stride 30          # 视频回放(无摄像头时验证)
python monitor.py --demo                                          # 合成帧自检,不依赖摄像头/YOLO

# 4. 查询故障
sqlite3 monitor.db "SELECT * FROM faults"
```

巡检模块输出：
- `monitor.db` 内 `observations` 表(每帧每灯 1 行,带 `state` / `state_change` 字段)
- `monitor.db` 内 `faults` 表(事件粒度,UPSERT 去重,记录 `start_ts`/`end_ts`/`fault_kind`)

判定规则：
- 夜见 `light_damage` 连续 3 帧 → 状态进 `FAULT`（写 `faults` 段, `fault_kind='night_damage'`）
- 白天见 `light_on` 连续 3 帧 → 状态进 `DAYLIGHT_ABNORMAL`（`fault_kind='day_light_on'`）
- 进入 / 退出 `FAULT` 都需连续 3 帧（对称防抖，树影/车灯短暂遮挡不会误切）
- OBB 边缘截断 → `cls=NULL` → 不进状态机，但 `observations.truncated=1` 留痕

`.gitignore` 已排除 `monitor.db` / `rois.json` / `*.pt` / `runs/`。
