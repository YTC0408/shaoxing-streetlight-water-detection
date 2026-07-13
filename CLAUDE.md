# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

绍兴大学校园场景的 YOLOv8 视觉检测项目，检测**路灯故障**（OBB 定向框）与**路面积水**（实例分割）。回复默认中文。

## 环境与命令

```bash
pip install ultralytics opencv-python Pillow      # tkinter 为 Python 内置
# 路灯训练/推理必须在 yolo conda 环境: D:\app\Python\anaconda\envs\yolo\python.exe
# (该环境有 CUDA, base 环境无 ultralytics, 会 ModuleNotFoundError)

# 训练(在仓库根目录执行, 脚本内路径相对根目录)
python merge_lights.py            # HBB 转 OBB + 合并 damaged_lights/pole_dst2328/new_light -> lights_merged
python augment_new_light.py       # 对 new_light 整杆 OBB 做翻转+光照增强(36 变体)
python train_lights_merged.py     # OBB 单权重, 75 轮, 早停+正则, 输出 runs/obb/lights_merged_obb/
python kfold_prep.py              # 生成 datasets/kfold/fold{k}.yaml (3 折)
python kfold_train.py             # 3-fold CV, 每折 75 轮, 汇总整杆 recall
python train.py                   # 仅积水分割(yolov8n-seg.pt)

# 推理
python detect_lights_wbf.py       # OBB 单模型摄像头推理, 画旋转四边形(WBF 已弃用, 仅保留单模型入口)
python test_ui.py                 # Tkinter GUI: 单/批量, 置信度滑块
python yolo-v.py                  # 积水推理(默认 yolov8n.pt)

# 巡检(路灯故障定时检测 + 状态记录)
python calibrate_rois.py --source 0 --output rois.json                  # 抓首帧,鼠标框选每盏灯
python calibrate_rois.py --auto --source 0 --output rois.json          # 自动 OBB 检测生成候选 + 人工微调
python monitor.py --rois rois.json                                     # 接摄像头,定时抓帧 + 状态机 + SQLite
python monitor.py --source <录好的mp4> --frame-stride 30              # 视频回放验证(无摄像头时)
python monitor.py --demo                                               # 合成帧自检(不依赖摄像头)
```

无测试框架。每个推理脚本入口有 `demo()` assert 自检, `main()` 启动时跑一次。

## 核心架构

### 双任务独立管线
1. **路灯故障**（OBB 定向框, YOLOv8-OBB）— 合并三数据集训单模型
2. **路面积水**（实例分割, 多边形）— `dataset_new`, `yolov8n-seg.pt`

### 类别语义对齐（关键约定）
夜间场景 **灯不亮 = 故障**。统一到 `light_damage`(=0, 故障/歪杆) / `light_on`(=1, 正常)：

| 数据集 | 原始类别 | 转换 | 合并后 ID |
|---|---|---|---|
| `damaged_lights` | `light_off`/`light_on` (原 Not Working/Working) | HBB → 轴对齐 4 点 OBB | 0/1 不变 |
| `pole_dst2328` | `lightening`/`damaged pole` | HBB → 轴对齐 4 点 OBB | 1/0 **翻转** |
| `new_light` | `light_damage`/`light_on` (Label Studio) | 真实斜 OBB, 原样保留 | 0/1 不变 |

- `merge_lights.py` 对 `pole_dst2328` 做 ID 翻转 remap (`{0:1, 1:0}`)，因 pole 原始 `0=lightening/1=damaged` 与统一约定相反。改合并逻辑时务必核对此翻转。
- `new_light` 提供真实倾斜四边形(歪杆), 这是识别整根歪杆的关键样本, **绝不要转 HBB**。
- 推理侧 `REMAP` 字典兼容旧命名 (`light_off`/`Not Working`/`damaged pole` → 0)，权重来源混杂也能正确显示。

### OBB 离线增强（`augment_new_light.py`）
8 个整杆 OBB 太少, 做翻转+光照扩到 36 变体。关键: 翻转要镜像 4 个点的 x 并交换左右点对以保持顶点绕序有效(详见脚本自检 `hflip_obb`)。

### 推理模块复用链
`detect_lights_wbf.py` 提供 OBB 单模型接口, `test_ui.py` 导入 `predict_frame_obb/is_truncated/EDGE_MARGIN/WEIGHTS_LIST/CONF`。改推理行为优先改 `detect_lights_wbf.py`，`test_ui.py` 自动继承。

### 边缘截断判定（uncertain）
`is_truncated()`：OBB 外接框触碰画面四边 (`EDGE_MARGIN` 像素内) 即判 **uncertain**(橙框), 不输出亮/灭结论——拍不全的灯不做故障判定。所有推理入口共用此逻辑。

### 巡检模块（monitor.py + calibrate_rois.py）
- **复用** `detect_lights_wbf.predict_frame_obb` / `is_truncated` / `REMAP`，不改推理侧。
- **ROI 锁定身份**: OBB 中心 ↔ ROI 中心最近邻匹配(半径 80px),每盏灯独立 tracker,多灯不混淆。
- **时间规则**: `monitor_config.json` 配 `night_start/end`(默认 18:00-06:00,跨午夜);文件不存在时回退默认。
- **故障规则**:
  - 夜见 `light_damage` 连续 3 帧 → 状态进 `FAULT`(写 `faults` 段)
  - 白天见 `light_on` 连续 3 帧 → 状态进 `DAYLIGHT_ABNORMAL`(写 `faults` 段)
  - 进入 / 退出 `FAULT` 都需连续 3 帧(对称防抖,树影/车灯不会误切)
- **截断** → `cls=NULL` → 不进状态机,但 `observations.truncated=1` 留痕。
- **存储**: SQLite `./monitor.db`,`observations`(每帧每灯 1 行) + `faults`(事件粒度,UPSERT 去重)。
- **标定流程**: 先 `calibrate_rois.py` 抓首帧鼠标框选 → `rois.json` → `monitor.py --rois rois.json` 启动。

## 数据集与权重（不入库）

`.gitignore` 排除所有图片/标签/`*.pt`/`runs/`, 仓库内**只保留各数据集的 `data.yaml`**。克隆后需自行放置数据集媒体, 或跑 `merge_lights.py` 重建 `lights_merged`。

- 数据集根: `datasets/{damaged_lights,dataset_new,lights_merged,pole_dst2328,new_light}/`
- 训练输出: `runs/obb/{name}/weights/best.pt`（推理脚本硬编码此路径）
- 整杆歪杆样本极少 (new_light 8 个 class0 + 增强 36 个), 训练后重点看 light_damage 单类 recall

## 约定

- 代码内 `# ponytail:` 注释标记的是有意为之的简化(如像素常量、简化算法), 不是待修 bug——保留意图注释。
- 提交信息用中文, 遵循 `feat/fix/refactor/docs` 前缀。
- `O:\下载` 之类的本地路径不要提交, 数据集媒体不入库。
