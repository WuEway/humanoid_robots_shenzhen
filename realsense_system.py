"""
RealSense传感器系统 - 面向对象设计
包含数据读取和处理的分离架构
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import threading
import time


class RGBDData:
    """RGBD数据容器类"""
    def __init__(self, color_image: Optional[np.ndarray] = None, 
                 depth_image: Optional[np.ndarray] = None,
                 color_frame: Optional[rs.frame] = None,
                 depth_frame: Optional[rs.frame] = None,
                 timestamp: Optional[float] = None):
        self.color_image = color_image
        self.depth_image = depth_image
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        self.timestamp = timestamp or time.time()
        
    def is_valid(self) -> bool:
        """检查数据是否有效"""
        return self.color_image is not None and self.depth_image is not None


class RealSenseDataReader(ABC):
    """RealSense数据读取器基类"""
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.is_running = False
        self.profile = None
        
    @abstractmethod
    def configure(self) -> None:
        """配置数据源"""
        pass
    
    def start(self) -> bool:
        """启动数据流"""
        try:
            self.configure()
            self.profile = self.pipeline.start(self.config)
            self.is_running = True
            print("数据流启动成功")
            return True
        except Exception as e:
            print(f"启动数据流失败: {e}")
            return False
    
    def stop(self) -> None:
        """停止数据流"""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("数据流已停止")
    
    def get_frame(self, timeout_ms: int = 1000) -> Optional[RGBDData]:
        """获取一帧数据"""
        if not self.is_running:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            frames = self.align.process(frames)
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame:
                return None
                
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            
            # 修复RGB/BGR通道顺序
            if color_image is not None:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            return RGBDData(
                color_image=color_image,
                depth_image=depth_image,
                color_frame=color_frame,
                depth_frame=depth_frame
            )
            
        except RuntimeError as e:
            print(f"获取帧失败: {e}")
            return None
    
    def get_camera_intrinsics(self) -> Optional[rs.intrinsics]:
        """获取相机内参"""
        if self.profile:
            color_stream = self.profile.get_stream(rs.stream.color)
            return color_stream.as_video_stream_profile().get_intrinsics()
        return None


class BagFileReader(RealSenseDataReader):
    """从bag文件读取数据"""
    
    def __init__(self, bag_file_path: str, repeat_playback: bool = False):
        super().__init__()
        self.bag_file_path = bag_file_path
        self.repeat_playback = repeat_playback
        self.playback = None
        
    def configure(self) -> None:
        """配置bag文件数据源"""
        self.config.enable_device_from_file(self.bag_file_path, self.repeat_playback)
        self.config.enable_all_streams()
        
    def start(self) -> bool:
        """启动bag文件播放"""
        if super().start():
            # 设置为非实时模式
            device = self.profile.get_device()
            self.playback = device.as_playback()
            self.playback.set_real_time(False)
            print(f"开始播放bag文件: {self.bag_file_path}")
            return True
        return False
    
    def set_real_time(self, real_time: bool) -> None:
        """设置是否实时播放"""
        if self.playback:
            self.playback.set_real_time(real_time)


class LiveStreamReader(RealSenseDataReader):
    """实时读取传感器数据"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        
    def configure(self) -> None:
        """配置实时数据流"""
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)


class RGBDProcessor(ABC):
    """RGBD数据处理器基类"""
    
    @abstractmethod
    def process(self, data: RGBDData) -> Any:
        """处理RGBD数据"""
        pass


class RGBDVisualizer(RGBDProcessor):
    """RGBD数据可视化处理器"""
    
    def __init__(self, window_name: str = "Color | Depth", 
                 depth_scale: float = 0.03):
        self.window_name = window_name
        self.depth_scale = depth_scale
        self.frame_count = 0
        
    def process(self, data: RGBDData) -> bool:
        """显示RGB和深度图像"""
        if not data.is_valid():
            return True
            
        self.frame_count += 1
        
        # 深度图转伪彩色
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(data.depth_image, alpha=self.depth_scale), 
            cv2.COLORMAP_JET
        )
        
        # 水平拼接显示
        if data.color_image is not None:
            # 确保两个图像高度一致
            if data.color_image.shape[:2] != depth_colormap.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, 
                                          (data.color_image.shape[1], data.color_image.shape[0]))
            stacked = np.hstack((data.color_image, depth_colormap))
        else:
            stacked = depth_colormap
            
        cv2.imshow(self.window_name, stacked)
        
        # 每30帧显示一次进度
        if self.frame_count % 30 == 0:
            print(f"已显示 {self.frame_count} 帧")
            
        # 检查键盘输入
        key = cv2.waitKey(1)
        if key & 0xFF in (27, ord("q")):  # ESC或q键退出
            return False
            
        return True
    
    def cleanup(self) -> None:
        """清理资源"""
        cv2.destroyAllWindows()



class RealSenseSystem:
    """RealSense系统主控制类"""
    
    def __init__(self, data_reader: RealSenseDataReader):
        self.data_reader = data_reader
        self.processors = []
        self.is_processing = False
        
    def add_processor(self, processor: RGBDProcessor) -> None:
        """添加数据处理器"""
        self.processors.append(processor)
        
    def start(self) -> bool:
        """启动系统"""
        if not self.data_reader.start():
            return False
            
        self.is_processing = True
        return True
        
    def stop(self) -> None:
        """停止系统"""
        self.is_processing = False
        self.data_reader.stop()
        
        # 清理所有处理器
        for processor in self.processors:
            if hasattr(processor, 'cleanup'):
                processor.cleanup()
                
    def run(self) -> None:
        """运行主循环"""
        if not self.start():
            print("系统启动失败")
            return
            
        try:
            while self.is_processing:
                # 获取数据
                data = self.data_reader.get_frame()
                if data is None:
                    break
                    
                # 处理数据
                for processor in self.processors:
                    try:
                        result = processor.process(data)
                        # 如果处理器返回False，表示要求停止
                        if result is False:
                            self.is_processing = False
                            break
                    except Exception as e:
                        print(f"处理器执行错误: {e}")
                        
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.stop()


if __name__ == "__main__":
    # 示例用法
    bag_file_path = "/home/yiwei/my_project/HumanoidRobots_shenzhen/takeout_bag.bag"
    
    # 创建bag文件读取器
    reader = BagFileReader(bag_file_path, repeat_playback=False)
    
    # 创建系统
    system = RealSenseSystem(reader)
    
    # 添加可视化处理器
    visualizer = RGBDVisualizer()
    system.add_processor(visualizer)
    
    # 添加GroundingDino处理器（使用grounding_dino_processor.py中的实现）
    # from grounding_dino_processor import AdvancedGroundingDinoProcessor
    # grounding_processor = AdvancedGroundingDinoProcessor()
    # system.add_processor(grounding_processor)
    
    # 运行系统
    system.run()
