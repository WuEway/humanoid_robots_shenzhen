"""
YOLO处理器的详细实现
包含目标检测和点云提取功能
集成YOLOv8模型进行实例分割
基于ROS2版本
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time
import math
from scipy.spatial.transform import Rotation as R

import rclpy
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

# ------------[本地模块导入]------------
from .handle_grasp_pose_estimation import HandleGraspEstimator
from .grasp_pose_estimator import GraspPoseEstimator
import open3d as o3d
from image_to_grasp.srv import ImageToGrasp
# ------------[结束本地模块导入]------------


import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs # 导入变换函数库
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException





class YOLOProcessor:
    """YOLO处理器 - 使用YOLOv8进行目标检测和分割"""
    
    def __init__(self, 
                 model_path: str = "non_ros_pkg/YOLO/weights/best.pt",
                 conf_threshold: float = 0.25,
                 imgsz: int = 640,
                 device: str = "cuda"):
        """
        初始化YOLO处理器
        
        Args:
            model_path: YOLO模型权重文件路径（相对于工作空间根目录）
            conf_threshold: 置信度阈值
            imgsz: 输入图像尺寸
            device: 计算设备 ("cuda" 或 "cpu")
        """
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        # 初始化torch设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型路径
        # 获取当前脚本文件的完整路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        self.model_path = os.path.join(workspace_root, model_path)

        # 初始化模型
        self.model = None
        
        # 帧计数器和时间跟踪
        self.last_detection_time = None
        self.detection_interval = 0.0

        # 加载模型
        self._load_models()
        
        print(f"YOLO处理器初始化完成，使用设备: {self.device}")
        
    def _load_models(self):
        """加载YOLO模型"""
        from ultralytics import YOLO

        # 加载YOLO模型
        print(f"正在加载YOLO模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("YOLO模型加载成功")

        # 检查 CUDA 可用性并移动到 GPU
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("✅ 模型已加载到 CUDA")
        else:
            print("⚠️ CUDA 不可用，使用 CPU 模式")

        # 打印每个权重的 device 信息（尽量兼容不同 YOLO 封装）
        # try:
        #     torch_module = getattr(self.model, 'model', self.model)
        #     if hasattr(torch_module, 'named_parameters'):
        #         for name, param in torch_module.named_parameters():
        #             print(f"权重: {name} -> device: {param.device}")
        # except Exception as e:
        #     print(f"打印权重设备信息时出错: {e}")
        
    def process(self, color_image: np.ndarray, depth_image: np.ndarray, camera_intrinsics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        处理RGBD数据, 检测目标并提取点云
        
        Args:
            color_image: RGB图像 (BGR格式)
            depth_image: 深度图像
            text_prompt: 检测目标的文本描述
            
        Returns:
            包含检测结果和点云的字典
        """
        if color_image is None or depth_image is None:
            return {"success": False, "error": "Invalid image data"}
            
        try:
            # 更新帧计数器和时间跟踪
            current_time = time.time()
            
            # 计算检测间隔时间
            if self.last_detection_time is not None:
                self.detection_interval = current_time - self.last_detection_time
            self.last_detection_time = current_time
            
            # 1. 目标检测
            detections = self._detect_objects(color_image)

            detect_time = time.time()
            print(f"YOLO检测耗时: {detect_time - current_time:.3f} 秒")
            
            # 2. 为每个检测结果提取点云
            point_clouds = []
            for detection in detections:
                point_cloud_data = self._extract_point_cloud(color_image, depth_image, detection, camera_intrinsics)
                if point_cloud_data is not None:
                    point_clouds.append({
                        "detection": detection,
                        "point_cloud": point_cloud_data
                    })
            
            extract_time = time.time()
            print(f"点云提取耗时: {extract_time - detect_time:.3f} 秒")
            # 3. 可视化结果
            result_image = self._visualize_detections(color_image, detections)

            finish_time = time.time()
            print(f"YOLO处理总耗时: {finish_time - current_time:.3f} 秒")

            return {
                "success": True,
                "detections": detections,
                "point_clouds": point_clouds,
                "result_image": result_image
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        使用YOLO检测目标物体
        
        Args:
            image: 输入RGB图像 (BGR格式，OpenCV标准)
            text_prompt: 目标描述文本（YOLO模式下此参数不使用，因为检测所有训练的类别）
            
        Returns:
            检测结果列表，包含边界框、置信度、标签和掩码
        """
        detections = []
        
        try:
            print("🤖 使用YOLO模型检测")
            
            self.imgsz = math.ceil(max(image.shape[:2]) / 32) * 32  # 确保是32的倍数
            # 使用YOLO进行推理
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False  # 禁用详细输出
            )
            
            # YOLO返回一个列表，我们只处理第一个结果
            if len(results) == 0:
                print("YOLO未检测到任何目标")
                return []
            
            result = results[0]
            
            # 检查是否有检测结果
            if result.masks is None or len(result.masks) == 0:
                print("YOLO未检测到任何带掩码的目标")
                return []
            
            print(f"YOLO检测到 {len(result.masks)} 个目标")
            
            # 获取掩码数据
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            
            # 获取原始图像尺寸
            orig_h, orig_w = result.masks.orig_shape
            mask_h, mask_w = masks_data.shape[1:]
            print(f"原始图像尺寸: {orig_w}x{orig_h}, 掩码尺寸: {mask_w}x{mask_h}")
            
            # 处理每个检测结果
            for i, (mask_padded, box, confidence, class_id) in enumerate(
                zip(masks_data, boxes, confidences, class_ids)
            ):
                # ------------ 手动调整掩码尺寸以匹配原始图像 ------------
                # 步骤1: 计算原始图像的宽高比
                orig_aspect = orig_w / orig_h
                # 步骤2: 计算 YOLO 缩放后的尺寸（保持宽高比）
                if orig_aspect >= 1:  # 宽图
                    scaled_w = self.imgsz
                    scaled_h = int(self.imgsz / orig_aspect)
                else:  # 高图
                    scaled_h = self.imgsz
                    scaled_w = int(self.imgsz * orig_aspect)
                # 步骤3: 计算 padding（YOLO 会将尺寸 pad 到最接近的 stride 倍数，通常是32）
                stride = 32
                padded_h = ((scaled_h + stride - 1) // stride) * stride
                padded_w = ((scaled_w + stride - 1) // stride) * stride
                # 步骤4: 去除 padding（裁剪到缩放后的尺寸）
                h_pad_total = padded_h - scaled_h
                w_pad_total = padded_w - scaled_w
                
                h_pad_top = h_pad_total // 2
                h_pad_bottom = h_pad_total - h_pad_top
                w_pad_left = w_pad_total // 2
                w_pad_right = w_pad_total - w_pad_left
                # 裁剪掉 padding
                if mask_h == padded_h and mask_w == padded_w:
                    # 掩码尺寸与预期的 padded 尺寸匹配
                    mask_unpadded = mask_padded[
                        h_pad_top:padded_h-h_pad_bottom,
                        w_pad_left:padded_w-w_pad_right
                    ]
                else:
                    # 如果不匹配，直接使用原始掩码
                    mask_unpadded = mask_padded
                print(f"去除 padding 后掩码形状: {mask_unpadded.shape}")
                # 步骤5: 现在 resize 到原始图像尺寸
                mask_resized = cv2.resize(mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                print(f"最终掩码形状: {mask_resized.shape}")

                # 二值化掩码
                binary_mask = (mask_resized > 0.001).astype(np.uint8)
                
                # 获取边界框坐标
                x1, y1, x2, y2 = box.astype(int)
                
                # 添加到检测结果
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                    "xyxy": [x1, y1, x2, y2],  # [x1, y1, x2, y2]
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "label": names[class_id] if class_id < len(names) else f"class_{class_id}",
                    "mask": binary_mask
                })
            
            # 每个类别只保留置信度最高的检测结果
            detections = self._pick_best_detection_per_class(detections)
            
            return detections
            
        except Exception as e:
            print(f"YOLO检测过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _pick_best_detection_per_class(self, detections: List[Dict]) -> List[Dict]:
        """每个类别只保留置信度最高的检测结果"""
        if not detections:
            return detections
        
        # 按类别标签分组（直接使用label而不是class_id）
        class_groups = {}
        for detection in detections:
            label = detection["label"]
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(detection)
        
        # 每个类别保留置信度最高的一个
        filtered_detections = []
        for label, group in class_groups.items():
            # 按置信度排序，取最高的
            best_detection = max(group, key=lambda x: x["confidence"])
            filtered_detections.append(best_detection)
            
            print(f"📦 类别 '{label}': 保留置信度最高的检测 ({best_detection['confidence']:.3f})")
        
        return filtered_detections
    
    def _extract_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray, detection: Dict, camera_intrinsics: Optional[Dict[str, float]] = None) -> Optional[Dict]:
        """
        根据检测结果提取目标物体的点云（包含颜色信息）
        
        Args:
            color_image: RGB图像 (BGR格式)
            depth_image: 深度图像
            detection: 检测结果
            camera_intrinsics: 相机内参字典
            
        Returns:
            包含点云和颜色信息的字典 {"points": (N, 3), "colors": (N, 3)} 或 None
        """
        try:
            mask = detection.get("mask")
            if mask is None: return None
            
            h, w = color_image.shape[:2]
            
            # 1. 准备内参
            if camera_intrinsics:
                fx = camera_intrinsics["fx"]
                fy = camera_intrinsics["fy"]
                cx = camera_intrinsics["cx"]
                cy = camera_intrinsics["cy"]
            else:
                fx, fy = 525.0, 525.0
                cx, cy = w / 2.0, h / 2.0

            points = []
            colors = []
            
            # # 遍历掩码区域 - 注意：np.where返回的是(y_coords, x_coords)
            # y_coords, x_coords = np.where(mask > 0)  # 这里是先y后x！
            
            # for y, x in zip(y_coords, x_coords):
            #     # 获取深度值
            #     depth = depth_image[y, x]  # 注意：深度图索引是[y, x]，即[行, 列]
            #     if depth > 0:  # 有效深度
            #         # 转换为3D坐标 (单位: 米，假设深度图单位为毫米)
            #         z = depth / 1000.0
            #         x_3d = (x - cx) * z / fx  # x对应列坐标
            #         y_3d = (y - cy) * z / fy  # y对应行坐标
                    
            #         points.append([x_3d, y_3d, z])
                    
            #         # 获取颜色 (BGR转RGB)
            #         b, g, r = color_image[y, x]  # 同样是[行, 列]索引
            #         colors.append([r, g, b])
            
            # if len(points) == 0:
            #     return None
            
            # points = np.array(points, dtype=np.float32)
            # colors = np.array(colors, dtype=np.uint8)
            
            # 2. 获取掩码区域的坐标索引 (Vectorized)
            # np.where 返回的是 (row_indices, col_indices)，即 (y, x)
            v_idx, u_idx = np.where(mask > 0)
            
            if len(v_idx) == 0:
                return None

            # 3. 批量获取深度值
            # 利用高级索引直接提取出所有掩码内的深度值
            z_raw = depth_image[v_idx, u_idx]
            
            # 4. 过滤无效深度 (深度为0的点)
            # 创建一个 boolean mask，只保留深度大于0的点
            valid_mask = z_raw > 0
            
            # 如果没有有效点，直接返回
            if not np.any(valid_mask):
                return None
                
            # 应用过滤：只保留有效的数据
            z_raw = z_raw[valid_mask]
            u = u_idx[valid_mask]
            v = v_idx[valid_mask]
            
            # 5. 核心矩阵计算 (Vectorized Math)
            # 将深度转换为米
            z = z_raw / 1000.0
            
            # 一次性计算所有点的 x 和 y
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # 6. 堆叠为 (N, 3) 数组
            # stack 按照最后一个维度合并，形成 [ [x1,y1,z1], [x2,y2,z2], ... ]
            points = np.stack([x, y, z], axis=-1).astype(np.float32)
            
            # 7. 提取并处理颜色
            # 同样利用索引提取颜色，并从 BGR 转为 RGB
            colors_bgr = color_image[v, u] # 注意这里是 v, u
            colors = colors_bgr[:, [2, 1, 0]].astype(np.uint8) # Swap BGR to RGB
            
            return {
                "points": points,
                "colors": colors
            }
            
        except Exception as e:
            print(f"点云提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """可视化检测结果"""
        result_image = image.copy()
        
        # 在图像顶部显示检测时间间隔
        if self.detection_interval > 0:
            fps = 1.0 / self.detection_interval if self.detection_interval > 0 else 0
            time_text = f"Detection Interval: {self.detection_interval:.3f}s ({fps:.1f} FPS)"
            cv2.putText(result_image, time_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            label = detection["label"]
            
            x, y, w, h = bbox
            
            # 使用不同颜色区分不同目标
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签和置信度
            text = f"{label}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y + text_size[1]), 
                         (x + text_size[0], y), color, -1)
            cv2.putText(result_image, text, (x, y + text_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制掩码（半透明）
            if "mask" in detection and detection["mask"] is not None:
                mask = detection["mask"]
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                    
                # 创建彩色掩码
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask > 0] = color
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        return result_image
    

    



class YOLOServer(Node):
    """基于ROS2的YOLO检测节点"""
    
    def __init__(self, 
                 node_name: str = "yolo_detector",
                 model_path: str = "non_ros_pkg/YOLO/weights/best.pt",
                 confidence_threshold: float = 0.25,
                 imgsz: int = 640,
                 camera_intrinsics: Optional[Dict[str, float]] = None,
                 enable_image_visualization: bool = True,
                 enable_pointcloud_visualization: bool = False,
                 target_class_name: Optional[str] = None):
        """
        初始化YOLO ROS2节点
        
        Args:
            node_name: 节点名称
            model_path: YOLO模型权重路径（相对于工作空间根目录）
            confidence_threshold: 置信度阈值
            imgsz: 输入图像尺寸
            camera_intrinsics: 相机内参 {"fx": 值, "fy": 值, "cx": 值, "cy": 值}
            enable_image_visualization: 是否显示检测结果的2D图像窗口
            enable_pointcloud_visualization: 是否显示抓取位姿计算中的3D点云窗口
            target_class_name: 用于点云发布和抓取位姿计算的目标类别名称
        """
        super().__init__(node_name)
        
        self.declare_parameter("edge_grasp_food_pos_frame", 'edge_grasp_food_pos')
        self.declare_parameter("handle_grasp_food_pos_frame", 'handle_grasp_food_pos')
        self.edge_grasp_food_pos_frame = self.get_parameter("edge_grasp_food_pos_frame").get_parameter_value().string_value
        self.handle_grasp_food_pos_frame = self.get_parameter("handle_grasp_food_pos_frame").get_parameter_value().string_value
        self.grasp_frame = self.edge_grasp_food_pos_frame  # 默认使用边缘抓取位姿框架
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 初始化处理器
        self.processor = YOLOProcessor(
            model_path=model_path,
            conf_threshold=confidence_threshold,
            imgsz=imgsz
        )
        self.get_logger().info("🤖 使用YOLO模型")

        # 初始化抓取位姿估计器
        pc_vis_status = "开启" if enable_pointcloud_visualization else "关闭"
        self.handle_grasp_estimator = HandleGraspEstimator(
            voxel_size=0.001,              # 提手点云内部处理的体素大小
            dbscan_eps=0.02,               # 提手聚类Eps
            dbscan_min_points=30,
            hsv_v_max=0.2,                # 黑色/深棕色的亮度阈值
            hsv_s_max=0.8,                 # 黑色/深棕色的饱和度阈值
            u_shape_min_points=500,         # U形簇最小点数
            u_shape_central_ratio=0.4,     # U形检测中心区域比例
            u_shape_hollow_ratio=0.10,     # U形空心比例
            grasp_bottom_height=0.03,      # 抓取点计算高度 (z_min + 0.03m)
            visualize=enable_pointcloud_visualization
        )
        self.edge_grasp_estimator = GraspPoseEstimator(visualize=enable_pointcloud_visualization)
        
        self.get_logger().info(f"🛠️  [Handle] U形提手抓取估计器已初始化 (3D点云可视化已{pc_vis_status})")

        # 定义坐标系名称，方便管理
        self.robot_base_frame = 'woosh_base_link'  # 确认这是你的机器人基座标系
        self.camera_frame = 'woosh_left_hand_rgbd_depth_optical_frame' # 确认这是你的相机坐标系

        # 初始化 TF2 Buffer 和 Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            
        # 配置参数
        self.target_label = target_class_name  # 直接使用目标类别名
        self.confidence_threshold = confidence_threshold
        self.enable_image_visualization = enable_image_visualization
        self.enable_pointcloud_visualization = enable_pointcloud_visualization
        
        # 目标类别设置
        if self.target_label:
            self.get_logger().info(f"☁️ 点云发布和抓取位姿计算目标已设置为: '{self.target_label}'")
        else:
            self.get_logger().warn(f"⚠️ 未设置目标类别名称，将不进行抓取位姿计算。")

        # 相机内参设置
        if camera_intrinsics is None:
            # 默认内参（需要根据实际相机调整）
            self.camera_intrinsics = {
                "fx": 848.0,  # 焦距x
                "fy": 480.0,  # 焦距y  
                "cx": 320.0,  # 主点x
                "cy": 240.0   # 主点y
            }
            self.get_logger().warn("⚠️  使用默认相机内参，建议传入实际内参")
        else:
            self.camera_intrinsics = camera_intrinsics
        self.has_camera_info = False  # 标志位，表示是否已收到相机内参
        self.declare_parameter('camera_info_topic', '/woosh/camera/woosh_left_hand_rgbd/color/camera_info')
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.get_logger().info(f"📷 相机内参: fx={self.camera_intrinsics['fx']}, fy={self.camera_intrinsics['fy']}")
        self.get_logger().info(f"📷 主点: cx={self.camera_intrinsics['cx']}, cy={self.camera_intrinsics['cy']}")
        
        # 处理状态
        self.frame_count = 0
        self.processing = False  # 新增：防止并发处理
        self.last_results = None
        
        # 检测结果存储 - 按类别分别存储
        self.detected_objects = {}  # 存储检测到的物体 {类别名: [名字, 置信度, 点云]}
        
        # 点云累积变量 (保持不变)
        self.accumulated_pcd = o3d.geometry.PointCloud()
        self.accumulation_voxel_size = 0.001
        self.target_detected_last_frame = False

        # 1. 创建 MessageFilter 订阅者
        # self.color_sub_filter = message_filters.Subscriber(
        #     self,
        #     Image,
        #     '/woosh/camera/woosh_left_hand_rgbd/color/image_raw'
        # )
        # self.depth_sub_filter = message_filters.Subscriber(
        #     self,
        #     Image,
        #     '/woosh/camera/woosh_left_hand_rgbd/aligned_depth_to_color/image_raw'
        # )

        self.grasp_service = self.create_service(
            ImageToGrasp,  # 替换为你的服务消息类型
            "/takeout_detection/image_to_grasp",  # 服务话题
            self.handle_grasp_request  # 服务回调函数
        )
        
        self.get_logger().info("✅ GroundingDino抓取服务端启动完成")
        self.get_logger().info(f"服务话题: /takeout_detection/image_to_grasp")
        self.get_logger().info(f"TF发布: {self.robot_base_frame} → {self.grasp_frame}")
    
        # 2. 创建时间同步器 (ApproximateTimeSynchronizer)
        # slop=0.1 表示允许 color 和 depth 之间有 0.1s (100ms) 的时间戳差异
        # self.ts = message_filters.ApproximateTimeSynchronizer(
        #     [self.color_sub_filter, self.depth_sub_filter],
        #     queue_size=10,  # 队列大小
        #     slop=0.1
        # )
        
        # 3. 注册同步后的回调函数
        # self.ts.registerCallback(self.synchronized_callback)

        # 创建抓取位姿发布者
        self.edge_grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/food_detection/edge_grasp_pose',
            10
        )
        self.handle_grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/food_detection/handle_grasp_pose',
            10
        )
        
        self.get_logger().info("🚀 YOLO ROS2节点启动完成")
        self.get_logger().info("📡 订阅话题:")
        self.get_logger().info("   RGB: /nbman/camera/nbman_head_rgbd/color/image_raw")
        self.get_logger().info("   深度: /nbman/camera/nbman_head_rgbd/aligned_depth_to_color/image_raw")
        self.get_logger().info("📤 发布话题:")
        self.get_logger().info("   抓取位姿: /grounding_dino/grasp_pose")
        self.get_logger().info("   调试点云: /grounding_dino/debug_pointcloud")
        if self.target_label:
            self.get_logger().info(f"🎯 检测目标类别: '{self.target_label}'")
        # self.get_logger().info(f"🎚️  置信度阈值: {self.confidence_threshold}")

    def camera_info_callback(self, msg: CameraInfo):
        """从camera_info获取真实的相机内参"""
        if not self.has_camera_info:
            K = msg.k  # 相机内参矩阵 (3x3)
            self.k_cam = np.array([
                [K[0], K[1], K[2]],
                [K[3], K[4], K[5]],
                [K[6], K[7], K[8]]
            ])
            self.has_camera_info = True
            
            self.get_logger().info(
                f"📷 Camera intrinsics received from camera_info: "
                f"fx={self.k_cam[0,0]:.2f}, fy={self.k_cam[1,1]:.2f}, "
                f"cx={self.k_cam[0,2]:.2f}, cy={self.k_cam[1,2]:.2f}"
            )
            self.cam_intrinsics = {
            "fx": self.k_cam[0,0],
            "fy": self.k_cam[1,1],
            "cx": self.k_cam[0,2],
            "cy": self.k_cam[1,2]
            }
            # 获取一次后可以取消订阅（可选）
            # self.destroy_subscription(self.camera_info_sub)

    def handle_grasp_request(self, request, response):
        """服务回调函数：处理客户端的图像请求，生成抓取位姿"""
        self.get_logger().info("📥 收到客户端图像请求，开始处理...")
        
        try:
            # 1. 解析客户端请求中的图像（RGB + 深度）
            # 转换RGB图像（sensor_msgs/Image → OpenCV）
            color_img = self.bridge.imgmsg_to_cv2(request.color_image, "bgr8")
            # 转换深度图像（假设深度图格式为16UC1，单位mm）
            depth_img = self.bridge.imgmsg_to_cv2(request.depth_image, "16UC1")
            
            # 2. 执行检测和点云提取（调用处理器）
            process_result = self.processor.process(
                color_image=color_img,
                depth_image=depth_img,
                # text_prompt=self.detect_prompt,
                camera_intrinsics=self.cam_intrinsics
            )
            
            if not process_result["success"]:
                response.success = False
                response.message = f"检测失败: {process_result.get('error', '未知错误')}"
                self.get_logger().error(response.message)
                return response
            
            # 3. 筛选目标点云（只保留目标标签的点云）
            target_pointcloud = None
            for pc_item in process_result["point_clouds"]:
                det_label = pc_item["detection"]["label"]
                det_conf = pc_item["detection"]["confidence"]
                if det_label == self.target_label and det_conf >= self.confidence_threshold:
                    target_pointcloud = pc_item["point_cloud"]
                    break
            
            if target_pointcloud is None:
                response.success = False
                response.message = f"未检测到目标标签: '{self.target_label}'（或置信度低于阈值）"
                self.get_logger().warn(response.message)
                return response
            
            # 4. 坐标转换：相机帧 → 机器人基座帧
            points_cam = target_pointcloud["points"]
            points_base = self._transform_point_cloud(points_cam, self.camera_frame, self.robot_base_frame)
            if points_base is None or len(points_base) == 0:
                response.success = False
                response.message = "点云坐标转换失败（相机→基座）"
                self.get_logger().error(response.message)
                return response
            # 5. 计算抓取位姿并且发布抓取位姿到TF和话题
            if request.grasp_type == "edge":
                self.grasp_frame = self.edge_grasp_food_pos_frame
                grasp_result = self.edge_grasp_estimator.calculate_grasp_pose(points_base, target_pointcloud["colors"])
                if not grasp_result:
                    response.success = False
                    response.message = "抓取位姿计算失败"
                    self.get_logger().error(response.message)
                    return response
                
                grasp_point, grasp_quat = grasp_result

                self._publish_grasp_tf(grasp_point, grasp_quat)
                self._publish_grasp_topic(grasp_point, grasp_quat, grasp_type="edge")
            elif request.grasp_type == "handle":
                self.grasp_frame = self.handle_grasp_food_pos_frame
                grasp_result = self.handle_grasp_estimator.calculate_grasp_pose(points_base, target_pointcloud["colors"])
                if not grasp_result:
                    response.success = False
                    response.message = "提手抓取位姿计算失败"
                    self.get_logger().error(response.message)
                    return response
                
                grasp_point, grasp_quat = grasp_result

                self._publish_grasp_tf(grasp_point, grasp_quat)
                self._publish_grasp_topic(grasp_point, grasp_quat, grasp_type="handle")
            else:
                response.success = False
                response.message = f"未知的抓取类型: '{request.grasp_type}'"
                self.get_logger().error(response.message)
                return response
            
            # 7. 构建服务响应（返回抓取位姿给客户端）
            response.success = True
            response.message = f"抓取{request.grasp_type}位姿生成成功, 已发布TF: {self.robot_base_frame} → {self.grasp_frame}"
            response.grasp_pose.position = grasp_point
            response.grasp_pose.orientation = grasp_quat
            
            self.get_logger().info(f"✅ 处理完成！{response.message}")
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"服务处理异常: {str(e)}"
            self.get_logger().error(f"❌ 服务异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return response

        
    def _transform_point_cloud(self, point_cloud_numpy: np.ndarray, source_frame: str, target_frame: str) -> Optional[np.ndarray]:
        """
        将一个NumPy点云从源坐标系转换到目标坐标系

        Args:
            point_cloud_numpy: (N, 3) 的NumPy数组
            source_frame: 源坐标系 (例如 'camera_color_optical_frame')
            target_frame: 目标坐标系 (例如 'base_link')

        Returns:
            转换后的 (N, 3) NumPy数组，如果失败则返回 None
        """
        if point_cloud_numpy.size == 0:
            return np.array([]) # 如果点云为空，直接返回空数组
        
        # self.get_logger().info(f"请求的时间戳: {timestamp.sec}.{timestamp.nanosec}")
        self.get_logger().info(f"当前时间戳: {self.get_clock().now().to_msg().sec}.{self.get_clock().now().to_msg().nanosec}")

        try:
            time_start = time.time()
            # 1. 查找指定时间戳的变换
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
            time_end = time.time()
            self.get_logger().info(f"🔄 查找变换耗时: {time_end - time_start:.3f} 秒")
            # transform = self.tf_buffer.lookup_transform(
            #     target_frame,
            #     source_frame,
            #     timestamp,  # <--- 使用传入的时间戳
            #     timeout=rclpy.duration.Duration(seconds=0.1) # 增加一个短暂超时
            # )

            # 2. 提取平移和旋转 (Scipy处理)
            t = transform.transform.translation
            translation = np.array([t.x, t.y, t.z])

            q = transform.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])

            # 3. 矩阵运算应用变换 (核心加速部分)
            time_start = time.time()
            # P_new = R * P_old + T
            transformed_points = rotation.apply(point_cloud_numpy) + translation
            
            time_end = time.time()
            self.get_logger().info(f"🔄 点云变换耗时(Vectorized): {time_end - time_start:.6f} 秒")
            
            return transformed_points.astype(np.float32)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"坐标变换失败: 从 '{source_frame}' 到 '{target_frame}': {e}")
            return None

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"坐标变换失败: 从 '{source_frame}' 到 '{target_frame}': {e}")
            return None

    def _publish_grasp_tf(self, grasp_point, grasp_quat):
        """发布抓取位姿到TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.robot_base_frame
        t.child_frame_id = self.grasp_frame
        # 位置
        t.transform.translation.x = grasp_point.x
        t.transform.translation.y = grasp_point.y
        t.transform.translation.z = grasp_point.z
        # 姿态（四元数）
        t.transform.rotation.x = grasp_quat.x
        t.transform.rotation.y = grasp_quat.y
        t.transform.rotation.z = grasp_quat.z
        t.transform.rotation.w = grasp_quat.w
        
        self.tf_broadcaster.sendTransform(t)

    def _publish_grasp_topic(self, grasp_point, grasp_quat, grasp_type):
        """发布抓取位姿到话题（可选，供其他节点订阅）"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.robot_base_frame
        pose_msg.pose.position = grasp_point
        pose_msg.pose.orientation = grasp_quat
        if grasp_type == "edge":
            self.edge_grasp_pose_pub.publish(pose_msg)

        if grasp_type == "handle":
            self.handle_grasp_pose_pub.publish(pose_msg)

    def cleanup(self):
        """清理资源"""
        if self.enable_image_visualization:
            cv2.destroyAllWindows()


def main():
    """主函数 - 启动ROS2节点"""
    rclpy.init()
    
    # 左手相机内参
    camera_intrinsics = {
        "fx": 608.837158203125,  # 实际焦距x
        "fy": 609.1549682617188,  # 实际焦距y
        "cx": 424.99688720703125,  # 实际主点x  
        "cy": 245.81431579589844   # 实际主点y
    }
    # 右手相机内参
    # camera_intrinsics = {
    #     "fx": 431.7814636230469,  # 实际焦距x
    #     "fy": 431.7814636230469,  # 实际焦距y
    #     "cx": 423.0641174316406,  # 实际主点x  
    #     "cy": 235.52688598632812   # 实际主点y
    # }
    
    # 创建YOLO检测节点
    node = YOLOServer(
        node_name="yolo_detector",
        model_path="non_ros_pkg/YOLO/weights/best.pt",  # YOLO模型路径
        confidence_threshold=0.25,  # YOLO置信度阈值
        imgsz=640,  # 输入图像尺寸
        camera_intrinsics=camera_intrinsics,
        enable_image_visualization=False,  # 设置为True可开启2D图像检测结果窗口
        enable_pointcloud_visualization=False, # 设置为True可开启3D点云处理窗口
        target_class_name="takeout bag"  # 设置你训练的YOLO模型中的目标类别名称
    )
    
    print("\n" + "="*60)
    print("🎯 YOLO + ROS2 检测系统")
    print("="*60)
    print("📡 ROS2话题:")
    print("  订阅 RGB: /nbman/camera/nbman_head_rgbd/color/image_raw")
    print("  订阅 深度: /nbman/camera/nbman_head_rgbd/aligned_depth_to_color/image_raw")
    print("  发布 抓取位姿: /grounding_dino/grasp_pose")
    print("="*60)
    print(f"ℹ️  2D图像可视化: {'启用' if node.enable_image_visualization else '禁用'}")
    print(f"ℹ️  3D点云可视化: {'启用' if node.enable_pointcloud_visualization else '禁用'}")
    print(f"ℹ️  抓取目标: '{node.target_label}'")
    print("="*60 + "\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 用户中断，程序退出")
    except Exception as e:
        node.get_logger().error(f"❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()