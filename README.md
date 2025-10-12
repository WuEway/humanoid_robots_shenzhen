# GroundingDino+SAM 外卖袋检测系统

## 项目概述

这是一个基于GroundingDino和SAM模型的实时目标检测与分割系统，专门用于检测外卖袋等物体，并提取相应的3D点云数据。系统采用面向对象架构，支持从RealSense bag文件或实时流读取RGB-D数据，自动检测目标、生成精确分割掩码、提取带颜色的3D点云，并通过增量式ICP配准维护全局一致的点云模型。

## 项目架构与数据流

系统由三个核心文件组成，数据流向如下：

```
[RealSense相机/Bag文件] 
         ↓ (RGB-D数据)
[realsense_system.py] ────> [数据读取与预处理]
         ↓ (对齐的RGB-D帧)
[grounding_dino_processor.py] ────> [目标检测+分割+点云提取]
         ↓ (检测结果+点云)
[文件系统] ────> [自动保存结果文件]
         ↓ (PLY点云文件)
[visualize_pointcloud.py] ────> [3D可视化显示]
```

### 核心文件功能
- **`realsense_system.py`**: 数据读取基础架构
- **`grounding_dino_processor.py`**: 主检测系统  
- **`visualize_pointcloud.py`**: 点云3D可视化工具

## 文件详细介绍

## 1. grounding_dino_processor.py - 主检测系统

核心检测文件，集成GroundingDino和SAM模型，实现目标检测、分割和3D点云提取。

### AdvancedGroundingDinoProcessor 类
- **功能**：集成GroundingDino和SAM模型的检测处理器
- **主要能力**：零样本目标检测、精确分割、3D点云提取、ICP配准
- **可调参数**：
  - `box_threshold=0.35`: 检测框置信度阈值
  - `text_threshold=0.25`: 文本匹配阈值  
  - `nms_threshold=0.8`: 非极大值抑制阈值
  - ICP参数: 最大迭代50次，收敛阈值1e-6

### InteractiveGroundingDinoVisualizer 类
- **功能**：交互式检测界面和自动保存管理
- **主要能力**：实时显示、键盘交互、自动保存检测结果
- **处理频率**：每30帧执行一次检测
- **自动保存**：检测到可信目标时直接保存，无需手动操作

## 2. realsense_system.py - 数据读取基础架构

提供统一的RGB-D数据访问接口，支持多种数据源。

### RealSenseDataReader 类（抽象基类）
- **功能**：定义数据读取器的统一接口规范
- **主要能力**：规范化不同数据源的访问方法

### BagFileReader 类
- **功能**：从RealSense bag文件读取预录制数据
- **主要能力**：bag文件播放、RGB-D对齐、帧序列控制
- **初始化参数**：`bag_file_path` - bag文件路径

### LiveStreamReader 类  
- **功能**：从实时RealSense相机读取数据流
- **主要能力**：实时数据获取、分辨率配置、帧率控制
- **可调参数**：
  - `width=640, height=480`: 图像分辨率
  - `fps=30`: 数据流帧率

### RGBDProcessor 类（抽象基类）
- **功能**：定义RGB-D数据处理器统一接口
- **主要能力**：为具体处理器提供标准化接口

### RGBDVisualizer 类
- **功能**：RGB-D数据处理器实例——可视化
- **主要能力**：实时显示RGB图像和深度图像、水平拼接显示、伪彩色深度图

### RealSenseSystem 类
- **功能**：RealSense系统主控制类，协调数据读取器和处理器
- **主要能力**：系统启动/停止、处理器添加、主循环控制、异常处理


## 3. visualize_pointcloud.py - 点云3D可视化工具

独立的点云可视化工具，支持多种格式的交互式3D显示。

### PointCloudVisualizer 类
- **功能**：提供点云文件的交互式3D可视化
- **主要能力**：多格式支持、文件选择对话框、3D交互显示
- **支持格式**：PLY（带/不带颜色）、PCD、XYZ
- **交互操作**：鼠标旋转/平移/缩放、键盘快捷键
- **命令行参数**：
  - `-i`: 交互式文件选择模式
  - `-d <目录>`: 指定浏览目录

## 使用指南

### 基本使用流程

#### 1. 运行主检测系统
```bash
python grounding_dino_processor.py
```

#### 2. 检测操作
- **程序启动**：自动寻找 `takeout_bag.bag` 文件作为数据源
- **输入检测目标**：程序启动时输入目标描述（如："pink takeout bag"）
- **预设快捷键**：
  - 按键 `1`：检测外卖袋 (takeout bag)
  - 按键 `2`：检测包 (bag)  
  - 按键 `3`：检测盒子 (box)
- **自动保存**：检测到可信目标时自动保存结果，无需手动操作
- **退出程序**：按键 `q` 或 `ESC`

#### 3. 查看3D点云结果
```bash
# 交互式选择文件
python visualize_pointcloud.py -i

# 指定目录浏览
python visualize_pointcloud.py -d detection_results/ -i

# 直接打开特定文件
python visualize_pointcloud.py detection_results/frame_0050_pink_takeout_bag/pink_takeout_bag_00_pointcloud_rgb.ply
```

#### 4. 查看bag中的RGB和深度图
```bash
# 可视化bag文件
python realsense_system.py
```

### 自动保存的文件说明

检测到可信目标时，系统自动在 `detection_results/frame_XXXX_目标名/` 创建文件（注意：所有文件名使用下划线而非空格）：

1. **`detection_overlay.jpg`**：
   - RGB原图叠加SAM分割掩码（彩色半透明）
   - GroundingDino检测框（绿色边框）
   - 置信度和目标标签信息

2. **`目标名_00_pointcloud.ply`**：
   - 无颜色版本的3D点云文件
   - 只包含XYZ空间坐标信息

3. **`目标名_00_pointcloud_rgb.ply`**（推荐）：
   - 带真实RGB颜色的3D点云文件
   - 包含XYZ坐标和RGB颜色信息

### 全局ICP配准文件

所有历史检测点云通过增量式ICP配准后统一保存：
- **`detection_results/global_registered_pointcloud.ply`**
- 包含所有检测目标的统一坐标系表示
- 每次新检测自动更新此文件

### 文件命名规范
```
detection_results/
├── frame_0001_pink_takeout_bag/           # 使用下划线连接
│   ├── detection_overlay.jpg
│   ├── pink_takeout_bag_00_pointcloud.ply
│   └── pink_takeout_bag_00_pointcloud_rgb.ply
├── frame_0002_red_bag/
│   ├── detection_overlay.jpg  
│   ├── red_bag_00_pointcloud.ply
│   └── red_bag_00_pointcloud_rgb.ply
└── global_registered_pointcloud.ply
```

## 参数配置详解

### 检测模型参数
在 `AdvancedGroundingDinoProcessor` 初始化时可配置：

```python
processor = AdvancedGroundingDinoProcessor(
    box_threshold=0.35,      # 检测框置信度阈值 (0.0-1.0)
    text_threshold=0.25,     # 文本匹配置信度阈值 (0.0-1.0)  
    nms_threshold=0.8,       # 非极大值抑制阈值 (0.0-1.0)
    device="cuda"            # 计算设备 ("cuda" 或 "cpu")
)
```

### ICP配准参数
在 `_perform_incremental_icp` 方法中配置：
- **最大迭代次数**: 50次 (可在代码中修改)
- **收敛阈值**: 1e-6 (可在代码中修改)
- **距离阈值**: 0.02米 (可在代码中修改)

### 处理频率参数
在 `InteractiveGroundingDinoVisualizer` 中：
- **检测频率**: 每30帧执行一次检测 (可在代码中修改)
- **自动保存**: 检测成功时立即保存

### 相机参数
在 `LiveStreamReader` 初始化时配置：
```python
reader = LiveStreamReader(
    width=640,    # 图像宽度
    height=480,   # 图像高度  
    fps=30        # 帧率
)
```

## 功能扩展与自定义

### 1. 添加新的检测目标
```python
# 在InteractiveGroundingDinoVisualizer的run方法中添加新的快捷键
elif key == ord('4'):
    prompt = "your new target"
    # 系统会自动检测并保存结果
```

### 2. 自定义数据源
```python
from realsense_system import RealSenseDataReader

class MyCustomReader(RealSenseDataReader):
    def get_next_frame(self):
        # 实现自定义数据读取逻辑
        return rgb_image, depth_image
        
    def get_camera_intrinsics(self):
        return camera_matrix

# 使用自定义数据源
visualizer = InteractiveGroundingDinoVisualizer()
visualizer.data_reader = MyCustomReader()
```

### 3. 调整检测敏感度
降低 `box_threshold` 可检测更多候选目标，但可能增加误检：
```python
# 更敏感的检测（可能有更多误检）
processor = AdvancedGroundingDinoProcessor(box_threshold=0.2)

# 更严格的检测（更少误检但可能漏检）  
processor = AdvancedGroundingDinoProcessor(box_threshold=0.5)
```

### 4. 自定义保存路径
```python
# 在_save_detection_results方法中修改保存路径
save_dir = f"custom_results/frame_{self.frame_counter:04d}_{clean_name}"
```

### 5. 批量处理模式
```python
def batch_process_bags(bag_files, target_prompt):
    for bag_file in bag_files:
        reader = BagFileReader(bag_file)
        processor = AdvancedGroundingDinoProcessor()
        # 自动处理每个bag文件
        while reader.has_more_frames():
            rgb, depth = reader.get_next_frame()
            if rgb is not None:
                result = processor.process_single_frame(rgb, depth, target_prompt)
                # 自动保存结果
```

## 环境配置与依赖

### 硬件要求
- **深度相机**：Intel RealSense D400系列深度相机
- **GPU加速**：NVIDIA GPU（用于GroundingDino和SAM模型推理）
- **内存要求**：至少8GB系统内存
- **存储空间**：至少2GB用于模型文件和检测结果

### 软件依赖
- **Python**：3.8+ 版本
- **深度学习框架**：PyTorch + CUDA支持
- **计算机视觉**：OpenCV-Python
- **相机接口**：pyrealsense2
- **3D处理**：Open3D (>=0.13.0)
- **AI模型**：GroundingDino、Segment-Anything

### 环境安装
```bash
# 激活推荐的conda环境
conda activate GroundingDINO

# 安装核心依赖
pip install open3d opencv-python pyrealsense2 numpy
```

## 使用建议与故障排除

### 最佳实践
1. **检测提示词**：使用具体描述，如"pink takeout bag"而非"bag"
2. **光照环境**：确保良好光照条件，有助于RGB颜色准确提取
3. **检测距离**：保持0.5-3米的合适检测距离
4. **存储管理**：注意定期清理`detection_results`目录

### 常见问题
- **模型加载失败**：检查CUDA环境和模型文件路径
- **相机连接问题**：确认RealSense相机正确连接
- **检测效果差**：调整检测阈值或改进提示词描述
- **ICP配准失败**：确保点云有足够几何特征

### 性能调优
- **检测频率**：系统每30帧处理一次，平衡精度与实时性
- **置信度阈值**：降低`box_threshold`可提高检测敏感度
- **存储优化**：大点云文件占用较多存储空间

## 系统优势

### 技术优势
1. **零样本检测**：无需训练即可检测任意文本描述的目标
2. **精确分割**：SAM提供像素级精确的目标分割
3. **真实颜色**：保持相机拍摄的真实RGB颜色信息
4. **空间一致性**：通过ICP配准保证多帧点云的空间统一
5. **模块化架构**：数据读取与处理完全分离，易于扩展
6. **自动化流程**：检测到目标即自动保存，无需手动操作

### 应用场景
- **机器人视觉**：外卖配送机器人的目标识别与定位
- **3D重建**：基于目标检测的选择性场景重建  
- **质量检测**：工业产品的外观检测与3D测量
- **增强现实**：实时目标检测与3D叠加显示

该系统特别适合需要结合目标检测与3D信息的应用场景，为外卖袋检测等实际应用提供了完整的解决方案。

