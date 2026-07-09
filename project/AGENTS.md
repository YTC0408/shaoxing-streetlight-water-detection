# AGENTS.md

> 本文件面向 AI 编程助手。阅读前请假设你对本项目一无所知。所有信息均基于项目实际文件内容整理，请勿臆测。

## 项目概览

本项目是**绍兴大学路灯故障与路面积水视觉检测**的实验性代码库，最初基于 Ultralytics YOLOv8 做通用目标检测演示，现已扩展为支持**大规模监控摄像头路灯状态巡检**的流水线。

- **目标场景**：绍兴大学校园道路及城市监控道路。
- **检测目标**：
  - 路灯故障：不亮、亮度异常、灯头缺失/破损、灯体位移/倾斜。
  - 路面积水（预留，需后续采集数据训练 YOLO 模型）。
- **核心能力**：
  - 基于 OpenCV 的**自动蒙板分割**（无需深度学习模型）。
  - 基于亮度与完整性的**多维度损伤检测**。
  - **时间感知**：区分昼夜，夜间做亮度分析，白天做结构完整性分析。
  - **日级批处理**：支持约 7000 路摄像头并行处理。
- **运行环境**：Ubuntu（部署端），开发端可用 Windows。

## 技术栈

- **语言**：Python 3.10。
- **核心依赖**：
  - `opencv-python`：视频读取、图像预处理、蒙板分割、特征提取。
  - `ultralytics`：YOLOv8 模型加载与推理（用于路面积水/通用目标检测演示）。
  - `pyyaml`：摄像头配置管理。
  - `numpy`：数值计算。
- **模型格式**：`.pt`（PyTorch 权重）。训练完成后可替换为自训练权重；`.pt`、`.pth`、`.onnx` 等模型文件已被 `.gitignore` 忽略，不纳入版本控制。

## 项目结构

```text
.
├── .gitignore              # 忽略模型权重、Python 缓存、虚拟环境、训练日志等
├── README.md               # 中文项目说明
├── AGENTS.md               # 本文件
├── requirements.txt        # 运行时依赖
├── yolo-v.py               # 原有实时摄像头/视频 YOLOv8 推理演示
├── auto_sync.py            # GitHub 自动同步脚本（可选）
├── auto_sync.bat           # Windows 下启动自动同步
├── run_daily.sh            # Ubuntu 每日巡检流水线入口
├── run_calibrate.sh        # Ubuntu 标定模式入口
├── config/
│   └── cameras.example.yaml    # 摄像头配置模板
├── src/
│   ├── preprocess.py       # OpenCV 自动蒙板分割
│   ├── detect_damage.py    # 亮度与完整性损伤检测
│   ├── calibrate.py        # 摄像头基准蒙板标定
│   ├── daily_pipeline.py   # 每日批处理流水线
│   ├── time_utils.py       # 时间感知工具
│   └── utils.py            # 通用工具
├── data/
│   ├── raw/                # 原始监控视频
│   ├── calibration/        # 基准蒙板与统计信息
│   ├── masks/              # 每日生成的蒙板
│   └── reports/            # 每日检测报告
└── logs/                   # 运行日志
```

## 环境安装（Ubuntu 部署端）

```bash
# 创建 conda 环境
conda create -n shaoxing-yolo python=3.10 -y
conda activate shaoxing-yolo

# 安装依赖
pip install -r requirements.txt
```

> 开发端若默认源较慢，可换清华镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

## 路灯巡检流水线使用方式

### 1. 准备数据

将监控视频按以下结构放置：

```text
data/raw/<camera_id>/<YYYY-MM-DD>/<video>.mp4
```

例如：

```text
data/raw/cam_001/2026-07-08/2026-07-08_20-00-00.mp4
```

### 2. 标定（首次运行）

选择每个摄像头**正常亮灯的夜晚视频**进行标定，生成基准蒙板：

```bash
conda activate shaoxing-yolo
bash run_calibrate.sh
```

或直接指定日期：

```bash
python src/daily_pipeline.py --date 2026-07-08 --calibrate --workers 8
```

标定结果保存在 `data/calibration/<camera_id>/`。

### 3. 每日巡检

```bash
conda activate shaoxing-yolo
bash run_daily.sh
```

`run_daily.sh` 默认处理前一天视频。也可手动指定：

```bash
python src/daily_pipeline.py --date 2026-07-08 --workers 16
```

### 4. 查看报告

报告输出在 `data/reports/<YYYY-MM-DD>/`：

- `summary.csv`：所有摄像头汇总表
- `summary.json`：JSON 格式汇总
- `<camera_id>.json`：单个摄像头详细结果

### 5. 定时任务（crontab）

在 Ubuntu 服务器上建议加入 crontab：

```bash
0 3 * * * /bin/bash /path/to/project/run_daily.sh >> /path/to/project/logs/cron.log 2>&1
```

## 检测逻辑说明

### 蒙板生成

- **标定阶段**：夜间高亮 Blob 检测 + 多帧共识，自动发现路灯位置。
- **日常阶段**：加载基准蒙板作为 ROI，夜间用亮度阈值、白天用自适应阈值提取当前灯头区域。

### 损伤判定

| 类型 | 指标 | 判定条件 |
|---|---|---|
| 不亮/严重变暗 | 蒙板内平均亮度 | 当前亮度 < 基准 50% |
| 亮度异常 | 蒙板内平均亮度 | 当前亮度 < 基准 70% |
| 灯头缺失/破损 | 蒙板面积 | 当前面积 < 基准 50% |
| 灯体位移/倾斜 | 蒙板中心偏移 | 偏移 > 15 像素 |

阈值可通过 `config/cameras.yaml` 调整。

## 扩展：路面积水检测

当前 `yolo-v.py` 演示了通用目标检测。若要训练路面积水检测模型：

1. 准备校园场景数据集，编写 Ultralytics 格式数据配置文件（如 `streetlight_water.yaml`）。
2. 训练：
   ```bash
   yolo train data=streetlight_water.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
3. 将最佳权重替换到推理脚本中。

## 代码风格与开发约定

- **文件命名**：脚本使用连字符或下划线命名（`yolo-v.py`、`daily_pipeline.py`）。
- **注释与文档**：README 与代码注释均使用中文。
- **变量命名**：模块内使用英文变量名，关键逻辑附中文注释。
- **路径处理**：使用 `pathlib.Path`，保证 Ubuntu/Windows 跨平台兼容。

## 安全与部署注意事项

- **不要在仓库中提交 `.pt`、`.pth`、`.onnx` 权重文件**：已被 `.gitignore` 排除。
- **不要提交监控视频**：`data/raw/` 应挂载外部存储或对象存储，不纳入版本控制。
- **敏感信息**：服务器地址、数据库连接等使用环境变量，并加入 `.gitignore`。
- **模型来源**：仅使用 Ultralytics 官方发布的权重或自行训练的权重。

## 常见修改点

| 需求 | 修改位置 |
|---|---|
| 调整路灯亮度阈值 | `config/cameras.yaml` → `night_brightness_threshold` |
| 调整破损判定面积比 | `config/cameras.yaml` → `integrity_area_ratio` |
| 调整并行进程数 | `run_daily.sh` 或命令行 `--workers` |
| 切换输入源（摄像头 ↔ 视频文件） | `yolo-v.py` 第 9 行 `cv2.VideoCapture(0)` |
| 更换检测模型 | `yolo-v.py` 第 5 行 `YOLO('yolov8n.pt')` |
