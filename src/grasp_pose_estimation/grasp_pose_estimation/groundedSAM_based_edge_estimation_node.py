"""
GroundingDino处理器的详细实现
包含目标检测和点云提取功能
集成真实的GroundingDino+SAM模型
基于ROS2版本
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from .grasp_pose_estimator import GraspPoseEstimator 


import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs # 导入变换函数库
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException





class AdvancedGroundingDinoProcessor:
    """高级GroundingDino处理器 - 集成真实的GroundingDino+SAM模型"""
    
    def __init__(self, 
                 grounding_dino_config_path: str = "non_ros_pkg/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint_path: str = "non_ros_pkg/Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
                 sam_encoder_version: str = "vit_h",
                 sam_checkpoint_path: str = "non_ros_pkg/Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 nms_threshold: float = 0.5,
                 device: str = "cuda"):
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        
        # 初始化torch设备
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        import os
        from pathlib import Path

        
        # 模型路径
        # 获取当前脚本文件的完整路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        self.grounding_dino_config_path = os.path.join(workspace_root, grounding_dino_config_path)
        self.grounding_dino_checkpoint_path = os.path.join(workspace_root, grounding_dino_checkpoint_path)
        self.sam_encoder_version = sam_encoder_version
        self.sam_checkpoint_path = os.path.join(workspace_root, sam_checkpoint_path)

        # 初始化模型
        self.grounding_dino_model = None
        self.sam_predictor = None
        
        # 帧计数器和时间跟踪
        self.last_detection_time = None
        self.detection_interval = 0.0

        # 延迟加载模型
        self._load_models()
        
        print(f"GroundingDino处理器初始化完成，使用设备: {self.device}")
        
    def _load_models(self):
        """加载GroundingDino和SAM模型"""
        # 导入深度学习库
        from groundingdino.util.inference import Model
        from segment_anything import sam_model_registry, SamPredictor
        
        # 加载GroundingDino模型
        print("正在加载GroundingDino模型...")
        self.grounding_dino_model = Model(
            model_config_path=self.grounding_dino_config_path,
            model_checkpoint_path=self.grounding_dino_checkpoint_path
        )
        print("GroundingDino模型加载成功")
        
        # 加载SAM模型
        print("正在加载SAM模型...")
        sam = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("SAM模型加载成功")
        
    def process(self, color_image: np.ndarray, depth_image: np.ndarray, text_prompt: str = "object", camera_intrinsics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
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
            detections = self._detect_objects(color_image, text_prompt)
            
            # 2. 为每个检测结果提取点云
            point_clouds = []
            for detection in detections:
                point_cloud_data = self._extract_point_cloud(color_image, depth_image, detection, camera_intrinsics)
                if point_cloud_data is not None:
                    point_clouds.append({
                        "detection": detection,
                        "point_cloud": point_cloud_data
                    })
            
            # 3. 可视化结果
            result_image = self._visualize_detections(color_image, detections)

            return {
                "success": True,
                "detections": detections,
                "point_clouds": point_clouds,
                "result_image": result_image
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_objects(self, image: np.ndarray, text_prompt: str) -> List[Dict]:
        """
        使用GroundingDino检测目标物体
        
        Args:
            image: 输入RGB图像 (BGR格式，OpenCV标准)
            text_prompt: 目标描述文本
            
        Returns:
            检测结果列表，包含边界框、置信度、标签和掩码
        """
        detections = []
        
        try:
            # 格式化文本提示（支持多个类别），将"."分隔的类别转换为列表
            if isinstance(text_prompt, str):
                classes = [c.strip() for c in text_prompt.split(".") if c.strip()]
            else:
                classes = text_prompt
            
            # GroundingDino检测（需要RGB格式）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用GroundingDino进行目标检测
            print("🤖 使用真实GroundingDino模型检测")
            detections_sv = self.grounding_dino_model.predict_with_classes(
                image=rgb_image,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            
            print(f"GroundingDino检测到 {len(detections_sv.xyxy)} 个目标")
            
            if len(detections_sv.xyxy) == 0:
                return []
            
            # NMS后处理
            import torch
            import torchvision
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections_sv.xyxy),
                torch.from_numpy(detections_sv.confidence),
                self.nms_threshold
            ).numpy().tolist()
            
            # 过滤检测结果
            filtered_boxes = detections_sv.xyxy[nms_idx]
            filtered_confidences = detections_sv.confidence[nms_idx]
            filtered_class_ids = detections_sv.class_id[nms_idx]
            
            print(f"NMS后保留 {len(filtered_boxes)} 个目标")
            
            # 使用SAM生成精确掩码
            masks = self._segment_with_sam(rgb_image, filtered_boxes)
            
            # 格式化检测结果
            all_detections = []
            for i, (box, confidence, class_id, mask) in enumerate(
                zip(filtered_boxes, filtered_confidences, filtered_class_ids, masks)
            ):
                x1, y1, x2, y2 = box.astype(int)
                all_detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                    "xyxy": [x1, y1, x2, y2],  # [x1, y1, x2, y2]
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "label": classes[class_id] if class_id < len(classes) else "object",
                    "mask": mask
                })
            
            # 每个类别只保留置信度最高的检测结果
            detections = self._pick_best_detection_per_class(all_detections)
            
            return detections
            
        except Exception as e:
            print(f"检测过程出错: {e}")
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

    def _segment_with_sam(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """使用SAM对检测框进行精确分割（来自SAM的例程）"""
        self.sam_predictor.set_image(image)
        result_masks = []
        
        for box in boxes:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # 选择得分最高的掩码
            index = np.argmax(scores)
            result_masks.append(masks[index])
            
        return result_masks
    
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
            # 必须使用SAM生成的mask，确保精确分割
            mask = detection.get("mask")
            if mask is None:
                print("⚠️  警告: 检测结果中没有mask，跳过点云提取")
                return None
            
            h, w = color_image.shape[:2]
            print(f"提取点云，图像尺寸: {w}x{h}")
            
            # 使用传入的相机内参或默认值
            if camera_intrinsics is not None:
                fx = camera_intrinsics["fx"]
                fy = camera_intrinsics["fy"]
                cx = camera_intrinsics["cx"]
                cy = camera_intrinsics["cy"]
            else:
                # 默认相机内参
                fx, fy = 525.0, 525.0  # 焦距
                cx, cy = w / 2.0, h / 2.0  # 光心
            
            points = []
            colors = []
            
            # 遍历掩码区域 - 注意：np.where返回的是(y_coords, x_coords)
            y_coords, x_coords = np.where(mask > 0)  # 这里是先y后x！
            
            for y, x in zip(y_coords, x_coords):
                # 获取深度值
                depth = depth_image[y, x]  # 注意：深度图索引是[y, x]，即[行, 列]
                if depth > 0:  # 有效深度
                    # 转换为3D坐标 (单位: 米，假设深度图单位为毫米)
                    z = depth / 1000.0
                    x_3d = (x - cx) * z / fx  # x对应列坐标
                    y_3d = (y - cy) * z / fy  # y对应行坐标
                    
                    points.append([x_3d, y_3d, z])
                    
                    # 获取颜色 (BGR转RGB)
                    b, g, r = color_image[y, x]  # 同样是[行, 列]索引
                    colors.append([r, g, b])
            
            if len(points) == 0:
                return None
            
            points = np.array(points, dtype=np.float32)
            colors = np.array(colors, dtype=np.uint8)

            return {
                "points": points,
                "colors": colors
            }
            
        except Exception as e:
            print(f"点云提取失败: {e}")
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
    

    



class GroundingDinoROS2Node(Node):
    """基于ROS2的GroundingDino检测节点"""
    
    def __init__(self, 
                 node_name: str = "grounding_dino_detector",
                 detection_prompt: str = "delivery box. pink takeout bag",
                 confidence_threshold: float = 0.4,
                 camera_intrinsics: Optional[Dict[str, float]] = None,
                 enable_image_visualization: bool = True,
                 enable_pointcloud_visualization: bool = False,
                 target_id_in_prompt: int = 1):
        """
        初始化GroundingDino ROS2节点
        
        Args:
            node_name: 节点名称
            detection_prompt: 检测目标的文本提示
            confidence_threshold: 置信度阈值
            camera_intrinsics: 相机内参 {"fx": 值, "fy": 值, "cx": 值, "cy": 值}
            enable_image_visualization: 是否显示检测结果的2D图像窗口
            enable_pointcloud_visualization: 是否显示抓取位姿计算中的3D点云窗口
            target_id_in_prompt: 点云发布目标在prompt中的索引
        """
        super().__init__(node_name)
        
        self.declare_parameter("grasp_food_pos_frame", 'grasp_food_pos')
        self.grasp_food_pos_frame = self.get_parameter("grasp_food_pos_frame").get_parameter_value().string_value
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 初始化处理器
        self.processor = AdvancedGroundingDinoProcessor()
        self.get_logger().info("🤖 使用GroundingDino+SAM模型")

        # 初始化抓取位姿估计器
        self.grasp_estimator = GraspPoseEstimator(visualize=enable_pointcloud_visualization)
        pc_vis_status = "开启" if enable_pointcloud_visualization else "关闭"
        self.get_logger().info(f"🛠️  抓取位姿估计器已初始化 (3D点云可视化已{pc_vis_status})")

        # 定义坐标系名称，方便管理
        self.robot_base_frame = 'nbman_base_link'  # 确认这是你的机器人基座标系
        self.camera_frame = 'nbman_head_rgbd_color_optical_frame' # 确认这是你的相机坐标系

        # 初始化 TF2 Buffer 和 Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            
        # 配置参数
        self.current_prompt = detection_prompt
        self.confidence_threshold = confidence_threshold
        self.enable_image_visualization = enable_image_visualization
        self.enable_pointcloud_visualization = enable_pointcloud_visualization
        
        # 解析用于点云发布的目标
        self.target_label = None
        prompt_classes = [c.strip() for c in self.current_prompt.split('.') if c.strip()]
        if 0 <= target_id_in_prompt < len(prompt_classes):
            self.target_label = prompt_classes[target_id_in_prompt]
            self.get_logger().info(f"☁️ 点云发布和抓取位姿计算目标已设置为: '{self.target_label}'")
        else:
            self.get_logger().warn(f"⚠️ 无效的 target_id_in_prompt: {target_id_in_prompt}。将不进行抓取位姿计算。")

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
            
        self.get_logger().info(f"📷 相机内参: fx={self.camera_intrinsics['fx']}, fy={self.camera_intrinsics['fy']}")
        self.get_logger().info(f"📷 主点: cx={self.camera_intrinsics['cx']}, cy={self.camera_intrinsics['cy']}")
        
        # 处理状态
        self.frame_count = 0
        self.processing = False  # 新增：防止并发处理
        self.last_results = None
        
        # 检测结果存储 - 按类别分别存储
        self.detected_objects = {}  # 存储检测到的物体 {类别名: [名字, 置信度, 点云]}
        
        # 图像数据缓存
        self.latest_color_image = None
        self.latest_depth_image = None
        
        # 创建订阅者
        self.color_sub = self.create_subscription(
            Image,
            '/woosh/camera/woosh_left_hand_rgbd/color/image_raw',
            self.color_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/woosh/camera/woosh_left_hand_rgbd/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # 创建抓取位姿发布者
        self.grasp_pose_pub = self.create_publisher(
            PoseStamped,
            '/grounding_dino/grasp_pose',
            10
        )
        
        # 创建调试点云发布者
        self.debug_pc_pub = self.create_publisher(
            PointCloud2,
            '/grounding_dino/debug_pointcloud',
            10
        )
        
        
        self.get_logger().info("🚀 GroundingDino ROS2节点启动完成")
        self.get_logger().info("📡 订阅话题:")
        self.get_logger().info("   RGB: /nbman/camera/nbman_head_rgbd/color/image_raw")
        self.get_logger().info("   深度: /nbman/camera/nbman_head_rgbd/aligned_depth_to_color/image_raw")
        self.get_logger().info("📤 发布话题:")
        self.get_logger().info("   抓取位姿: /grounding_dino/grasp_pose")
        self.get_logger().info("   调试点云: /grounding_dino/debug_pointcloud")
        self.get_logger().info(f"🎯 检测目标: '{self.current_prompt}'")
        self.get_logger().info(f"🎚️  置信度阈值: {self.confidence_threshold}")
    
        
    def color_callback(self, msg: Image):
        """RGB图像回调函数"""
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_if_ready(self.latest_color_image, self.latest_depth_image)
            self.get_logger().info("✅ 进入color_callback")
                            
        except Exception as e:
            self.get_logger().error(f"RGB图像处理失败: {e}")

    def depth_callback(self, msg: Image):
        """深度图像回调函数"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.process_if_ready(self.latest_color_image, self.latest_depth_image)
            self.get_logger().info("✅ 进入depth_callback")
                 
        except Exception as e:
            self.get_logger().error(f"深度图像处理失败: {e}")
    
    def process_if_ready(self, cur_color_image: np.ndarray, cur_depth_image: np.ndarray):
        """检查是否有完整的RGBD数据，如果有则处理"""
        if cur_color_image is None or cur_depth_image is None:
            return
        
        # 防止并发处理
        if self.processing:
            return
            
        self.frame_count += 1
        
        # 根据配置决定处理频率
        self.processing = True  # 设置处理标志
            
        try:
            self.get_logger().info(f"第 {self.frame_count} 帧 - 检测目标: '{self.current_prompt}'")
            
            # 执行检测
            self.last_results = self.processor.process(
                cur_color_image, 
                cur_depth_image, 
                self.current_prompt,
                camera_intrinsics=self.camera_intrinsics
            )
            
            if self.last_results["success"]:
                # 可选地显示结果图像
                if self.enable_image_visualization:
                    result_image = self.last_results["result_image"]
                    cv2.imshow("GroundingDino Results", result_image)
                    cv2.waitKey(1)

                # 更新检测结果
                self._update_detection_results()
                
                # 打印检测信息
                detections = self.last_results["detections"]
                if detections:
                    self.get_logger().info(f"✅ 检测到 {len(detections)} 个目标:")
                    for i, det in enumerate(detections):
                        self.get_logger().info(f"  {i+1}. {det['label']}: {det['confidence']:.3f}")
                        # 打印点云信息
                        point_clouds = self.last_results["point_clouds"]
                        if i < len(point_clouds) and point_clouds[i]["point_cloud"] is not None:
                            pc_size = len(point_clouds[i]["point_cloud"]["points"])
                            self.get_logger().info(f"     点云大小: {pc_size} 个点")
                else:
                    self.get_logger().warn(f"❌ 未检测到目标: '{self.current_prompt}'")
            else:
                if self.enable_image_visualization:
                    cv2.imshow("GroundingDino Results", cur_color_image)
                    cv2.waitKey(1)
                self.get_logger().error(f"❌ 检测失败: {self.last_results.get('error', 'Unknown error')}")
                self.last_results = None
                
        finally:
            self.processing = False  # 清除处理标志
            # 将检测结果点云转换到机器人坐标系下，计算抓取点
            if self.target_label and self.target_label in self.detected_objects:
                label, _, pointcloud_dict = self.detected_objects[self.target_label]
                
                if pointcloud_dict is not None and len(pointcloud_dict["points"]) > 0:
                    points_cam = pointcloud_dict["points"]
                    colors_cam = pointcloud_dict["colors"]
                    
                    self.get_logger().info(f"正在为 '{label}' 计算抓取位姿...")
                    
                    # 将点云转换到机器人基座标系
                    points_robot = self._transform_point_cloud(points_cam, self.camera_frame, self.robot_base_frame)

                    # 检查转换是否成功
                    if points_robot is not None and points_robot.shape[0] > 0:
                        # # 创建并发布调试用的点云消息
                        # debug_pc_msg = self._create_point_cloud_msg(points_robot, colors_cam, self.robot_base_frame)
                        # self.debug_pc_pub.publish(debug_pc_msg)
                        # self.get_logger().info("已发布调试点云到 /grounding_dino/debug_pointcloud")

                        # 使用GraspPoseEstimator计算抓取位姿
                        grasp_pose_result = self.grasp_estimator.calculate_grasp_pose(points_robot, colors_cam)
                        
                        if grasp_pose_result:
                            grasp_point, grasp_orientation = grasp_pose_result
                            
                            # 创建并发布PoseStamped消息
                            pose_msg = PoseStamped()
                            pose_msg.header.stamp = self.get_clock().now().to_msg()
                            pose_msg.header.frame_id = self.robot_base_frame
                            pose_msg.pose.position = grasp_point
                            pose_msg.pose.orientation = grasp_orientation
                            
                            self.grasp_pose_pub.publish(pose_msg)
                            
                            # 发布至TF
                            t = TransformStamped()
                            t.header.stamp = self.get_clock().now().to_msg()
                            t.header.frame_id = self.robot_base_frame
                            t.child_frame_id = self.grasp_food_pos_frame
                            t.transform.translation.x = grasp_point.x
                            t.transform.translation.y = grasp_point.y
                            t.transform.translation.z = grasp_point.z
                            t.transform.rotation.x = grasp_orientation.x
                            t.transform.rotation.y = grasp_orientation.y
                            t.transform.rotation.z = grasp_orientation.z
                            t.transform.rotation.w = grasp_orientation.w

                            self.tf_broadcaster.sendTransform(t)
                            self.get_logger().info(f"✅ 已发布TF变换: {self.robot_base_frame} -> {self.grasp_food_pos_frame}")
                            self.get_logger().info(f"✅ 成功发布抓取位姿到话题 '{self.grasp_pose_pub.topic}'")
                    else:
                        self.get_logger().warn("点云坐标转换失败或结果为空，跳过抓取计算")

    def _update_detection_results(self):
        """更新检测结果到成员变量"""
        if not self.last_results or not self.last_results.get("success"):
            return
            
        detections = self.last_results.get("detections", [])
        point_clouds = self.last_results.get("point_clouds", [])
        
        # 直接处理每个检测结果（已经是每类最佳的了）
        for i, detection in enumerate(detections):
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)
            
            # 只更新置信度超过阈值且有点云的检测
            if (confidence >= self.confidence_threshold and 
                i < len(point_clouds) and 
                point_clouds[i]["point_cloud"] is not None):
                
                # 直接更新该类别的检测结果
                self.detected_objects[label] = [
                    label,
                    confidence,
                    point_clouds[i]["point_cloud"]
                ]
                
                self.get_logger().info(f"🔄 更新类别 '{label}': 置信度 {confidence:.3f}")
            else:
                self.get_logger().debug(f" 跳过 '{label}': 置信度 {confidence:.3f} < {self.confidence_threshold} 或无点云")
        
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
            
        try:
            # 1. 查找最新的可用变换
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())

            # 2. 逐点进行变换
            # (对于大规模点云有更高效的方法，但这种方法最清晰、最可靠)
            transformed_points = []
            for point in point_cloud_numpy:
                # 将NumPy点封装成PointStamped消息
                p_stamped = PointStamped()
                p_stamped.header.frame_id = source_frame
                p_stamped.point.x = float(point[0])
                p_stamped.point.y = float(point[1])
                p_stamped.point.z = float(point[2])

                # 应用变换
                p_transformed = tf2_geometry_msgs.do_transform_point(p_stamped, transform)
                
                transformed_points.append([
                    p_transformed.point.x,
                    p_transformed.point.y,
                    p_transformed.point.z
                ])
            
            return np.array(transformed_points, dtype=np.float32)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"坐标变换失败: 从 '{source_frame}' 到 '{target_frame}': {e}")
            return None

    def _create_point_cloud_msg(self, points: np.ndarray, colors: np.ndarray, frame_id: str) -> PointCloud2:
        """
        根据点和颜色数据创建PointCloud2消息
        """
        # 2. 创建一个 Header 对象
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # 将颜色(R,G,B)合并到一个UINT32字段中
        colors_bgr = colors[:, [2, 1, 0]]
        cv2.imshow("Colors BGR", colors_bgr)
        rgb_packed = np.array((colors_bgr[:, 2] << 16) | (colors_bgr[:, 1] << 8) | (colors_bgr[:, 0]), dtype=np.uint32)
        
        # 将点和颜色数据合并
        # 创建一个结构化数组
        point_data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        point_data['x'] = points[:, 0]
        point_data['y'] = points[:, 1]
        point_data['z'] = points[:, 2]
        point_data['rgb'] = rgb_packed

        # 创建PointCloud2消息
        pc_msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=16, # 4 (x) + 4 (y) + 4 (z) + 4 (rgb)
            row_step=16 * points.shape[0],
            data=point_data.tobytes()
        )
        pc_msg.header.frame_id = frame_id
        
        return pc_msg

    def cleanup(self):
        """清理资源"""
        if self.enable_image_visualization:
            cv2.destroyAllWindows()


def main():
    """主函数 - 启动ROS2节点"""
    rclpy.init()
    
    # 实际的相机内参（需要根据您的相机调整）
    camera_intrinsics = {
        "fx": 427.8312,  # 实际焦距x
        "fy": 427.3405,  # 实际焦距y
        "cx": 430.8444,  # 实际主点x  
        "cy": 246.7171   # 实际主点y
    }
    
    # 创建GroundingDino检测节点
    node = GroundingDinoROS2Node(
        node_name="grounding_dino_detector",
        detection_prompt="delivery box. pink takeout bag",
        confidence_threshold=0.4,
        camera_intrinsics=camera_intrinsics,
        enable_image_visualization=True,  # 设置为True可开启2D图像检测结果窗口
        enable_pointcloud_visualization=False, # 设置为True可开启3D点云处理窗口
        target_id_in_prompt=1  # 0是'delivery box', 1是'pink takeout bag'
    )
    
    print("\n" + "="*60)
    print("🎯 GroundingDino + ROS2 检测系统")
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