import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from image_to_grasp.srv import ImageToGrasp
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
# from .grounding_dino_processor import AdvancedGroundingDinoProcessor
# # 导入原有处理器和抓取位姿估计器
from .grasp_pose_estimator import GraspPoseEstimator 


class AdvancedGroundingDinoProcessor:
    """保持原有处理器逻辑不变，负责模型加载、检测和点云提取"""
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
        
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型路径处理
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        self.grounding_dino_config_path = os.path.join(workspace_root, grounding_dino_config_path)
        self.grounding_dino_checkpoint_path = os.path.join(workspace_root, grounding_dino_checkpoint_path)
        self.sam_encoder_version = sam_encoder_version
        self.sam_checkpoint_path = os.path.join(workspace_root, sam_checkpoint_path)

        # 模型初始化
        self.grounding_dino_model = None
        self.sam_predictor = None
        self._load_models()
        
        print(f"GroundingDino处理器初始化完成，使用设备: {self.device}")
        
    def _load_models(self):
        from groundingdino.util.inference import Model
        from segment_anything import sam_model_registry, SamPredictor
        
        print("正在加载GroundingDino模型...")
        self.grounding_dino_model = Model(
            model_config_path=self.grounding_dino_config_path,
            model_checkpoint_path=self.grounding_dino_checkpoint_path
        )
        print("GroundingDino模型加载成功")
        
        print("正在加载SAM模型...")
        sam = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("SAM模型加载成功")
        
    def process(self, color_image: np.ndarray, depth_image: np.ndarray, text_prompt: str = "object", camera_intrinsics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if color_image is None or depth_image is None:
            return {"success": False, "error": "Invalid image data"}
            
        try:
            # 1. 目标检测
            detections = self._detect_objects(color_image, text_prompt)
            
            # 2. 提取点云
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
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_objects(self, image: np.ndarray, text_prompt: str) -> List[Dict]:
        detections = []
        try:
            # 处理文本提示
            classes = [c.strip() for c in text_prompt.split(".") if c.strip()] if isinstance(text_prompt, str) else text_prompt
            if not classes:
                return []
            
            # GroundingDino检测
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections_sv = self.grounding_dino_model.predict_with_classes(
                image=rgb_image,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            
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
            
            filtered_boxes = detections_sv.xyxy[nms_idx]
            filtered_confidences = detections_sv.confidence[nms_idx]
            filtered_class_ids = detections_sv.class_id[nms_idx]
            
            # SAM分割
            masks = self._segment_with_sam(rgb_image, filtered_boxes)
            
            # 格式化结果
            all_detections = []
            for i, (box, confidence, class_id, mask) in enumerate(
                zip(filtered_boxes, filtered_confidences, filtered_class_ids, masks)
            ):
                x1, y1, x2, y2 = box.astype(int)
                all_detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "xyxy": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "label": classes[class_id] if class_id < len(classes) else "object",
                    "mask": mask
                })
            
            # 保留每类最高置信度结果
            detections = self._pick_best_detection_per_class(all_detections)
            return detections
            
        except Exception as e:
            print(f"检测过程出错: {e}")
            return []
    
    def _pick_best_detection_per_class(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return detections
        
        class_groups = {}
        for detection in detections:
            label = detection["label"]
            class_groups[label] = class_groups.get(label, []) + [detection]
        
        filtered_detections = []
        for label, group in class_groups.items():
            best_detection = max(group, key=lambda x: x["confidence"])
            filtered_detections.append(best_detection)
        
        return filtered_detections

    def _segment_with_sam(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in boxes:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            result_masks.append(masks[np.argmax(scores)])
        return result_masks
    
    def _extract_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray, detection: Dict, camera_intrinsics: Optional[Dict[str, float]] = None) -> Optional[Dict]:
        try:
            mask = detection.get("mask")
            if mask is None:
                return None
            
            h, w = color_image.shape[:2]
            # 相机内参处理
            if camera_intrinsics:
                fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
                cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
            else:
                fx, fy = 525.0, 525.0
                cx, cy = w / 2.0, h / 2.0
            
            # 提取掩码区域坐标
            y_coords, x_coords = np.where(mask > 0)
            points = []
            colors = []
            
            for y, x in zip(y_coords, x_coords):
                depth = depth_image[y, x]
                if depth <= 0:
                    continue
                # 像素→相机3D坐标转换
                z = depth / 1000.0  # 深度图单位假设为mm
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                
                points.append([x_3d, y_3d, z])
                # 颜色转换（BGR→RGB）
                b, g, r = color_image[y, x]
                colors.append([r, g, b])
            
            if len(points) == 0:
                return None
            
            return {
                "points": np.array(points, dtype=np.float32),
                "colors": np.array(colors, dtype=np.uint8)
            }
            
        except Exception as e:
            print(f"点云提取失败: {e}")
            return None
    
    def _visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        result_image = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            label = detection["label"]
            x, y, w, h = bbox
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            # 绘制标签
            text = f"{label}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 绘制掩码
            if "mask" in detection and detection["mask"] is not None:
                mask = detection["mask"].astype(np.uint8) * 255
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask > 0] = color
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        return result_image


class GroundingDinoGraspServer(Node):
    """ROS 2 服务端:接收图像请求, 返回抓取位姿并发布TF"""
    def __init__(self):
        super().__init__("grounding_dino_grasp_server")
        
        # 1. 声明参数（可通过启动文件或命令行修改）
        self.declare_parameter("grasp_food_pos_frame", "grasp_food_pos")
        self.declare_parameter("detection_prompt", "delivery box. pink takeout bag")
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("target_id_in_prompt", 1)  # 1对应"pink takeout bag"
        self.declare_parameter("robot_base_frame", "woosh_base_link")
        # self.declare_parameter("camera_frame", "woosh_head_rgbd_color_optical_frame")
        self.declare_parameter("camera_frame", "woosh_left_hand_rgbd_color_optical_frame")
        
        # 获取参数
        self.grasp_frame = self.get_parameter("grasp_food_pos_frame").get_parameter_value().string_value
        self.detect_prompt = self.get_parameter("detection_prompt").get_parameter_value().string_value
        self.conf_thresh = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.target_id = self.get_parameter("target_id_in_prompt").get_parameter_value().integer_value
        self.base_frame = self.get_parameter("robot_base_frame").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        
        # 2. 初始化核心组件
        self.bridge = CvBridge()
        # 初始化处理器（加载GroundingDino+SAM模型）
        self.processor = AdvancedGroundingDinoProcessor()
        # 初始化抓取位姿估计器
        self.grasp_estimator = GraspPoseEstimator(visualize=False)
        # 初始化TF（用于坐标转换和发布）
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # 初始化抓取位姿发布者（可选，用于话题发布）
        self.grasp_pub = self.create_publisher(PoseStamped, "/grounding_dino/grasp_pose", 10)
        
        # 3. 解析目标标签（从检测提示中提取）
        self.target_label = None
        prompt_classes = [c.strip() for c in self.detect_prompt.split(".") if c.strip()]
        if 0 <= self.target_id < len(prompt_classes):
            self.target_label = prompt_classes[self.target_id]
            self.get_logger().info(f"目标抓取标签: '{self.target_label}'")
        else:
            self.get_logger().error(f"无效target_id_in_prompt: {self.target_id}，服务将无法正常工作")
            return
        
        # 4. 相机内参（根据实际相机调整）
        self.cam_intrinsics = {
            "fx": 427.8312,
            "fy": 427.3405,
            "cx": 430.8444,
            "cy": 246.7171
        }
        self.get_logger().info(f"相机内参: fx={self.cam_intrinsics['fx']}, fy={self.cam_intrinsics['fy']}, cx={self.cam_intrinsics['cx']}, cy={self.cam_intrinsics['cy']}")
        
        # 5. 创建服务（服务类型：ImageToPose）
        self.grasp_service = self.create_service(
            ImageToGrasp,  # 替换为你的服务消息类型
            "/grounding_dino/image_to_grasp",  # 服务话题
            self.handle_grasp_request  # 服务回调函数
        )
        
        self.get_logger().info("✅ GroundingDino抓取服务端启动完成")
        self.get_logger().info(f"服务话题: /grounding_dino/image_to_grasp")
        self.get_logger().info(f"TF发布: {self.base_frame} → {self.grasp_frame}")

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
                text_prompt=self.detect_prompt,
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
                if det_label == self.target_label and det_conf >= self.conf_thresh:
                    target_pointcloud = pc_item["point_cloud"]
                    break
            
            if target_pointcloud is None:
                response.success = False
                response.message = f"未检测到目标标签: '{self.target_label}'（或置信度低于阈值）"
                self.get_logger().warn(response.message)
                return response
            
            # 4. 坐标转换：相机帧 → 机器人基座帧
            points_cam = target_pointcloud["points"]
            points_base = self._transform_point_cloud(points_cam, self.camera_frame, self.base_frame)
            if points_base is None or len(points_base) == 0:
                response.success = False
                response.message = "点云坐标转换失败（相机→基座）"
                self.get_logger().error(response.message)
                return response
            
            # 5. 计算抓取位姿
            grasp_result = self.grasp_estimator.calculate_grasp_pose(points_base, target_pointcloud["colors"])
            if not grasp_result:
                response.success = False
                response.message = "抓取位姿计算失败"
                self.get_logger().error(response.message)
                return response
            
            grasp_point, grasp_quat = grasp_result
            
            # 6. 发布抓取位姿到TF和话题
            self._publish_grasp_tf(grasp_point, grasp_quat)
            self._publish_grasp_topic(grasp_point, grasp_quat)
            
            # 7. 构建服务响应（返回抓取位姿给客户端）
            response.success = True
            response.message = f"抓取位姿生成成功, 已发布TF: {self.base_frame} → {self.grasp_frame}"
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

    def _transform_point_cloud(self, points_cam: np.ndarray, source_frame: str, target_frame: str) -> Optional[np.ndarray]:
        """将相机帧点云转换到机器人基座帧"""
        if points_cam.size == 0:
            return None
        
        try:
            # 获取相机→基座的TF变换
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()  # 获取最新变换
            )
            
            # 逐点转换（适合中小规模点云）
            transformed_points = []
            for point in points_cam:
                p_stamped = PointStamped()
                p_stamped.header.frame_id = source_frame
                p_stamped.point.x = float(point[0])
                p_stamped.point.y = float(point[1])
                p_stamped.point.z = float(point[2])
                
                # 应用TF变换
                p_trans = tf2_geometry_msgs.do_transform_point(p_stamped, transform)
                transformed_points.append([p_trans.point.x, p_trans.point.y, p_trans.point.z])
            
            return np.array(transformed_points, dtype=np.float32)
        
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF转换失败: {source_frame} → {target_frame}: {str(e)}")
            return None

    def _publish_grasp_tf(self, grasp_point, grasp_quat):
        """发布抓取位姿到TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
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

    def _publish_grasp_topic(self, grasp_point, grasp_quat):
        """发布抓取位姿到话题（可选，供其他节点订阅）"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.pose.position = grasp_point
        pose_msg.pose.orientation = grasp_quat
        
        self.grasp_pub.publish(pose_msg)


def main():
    rclpy.init()
    # 创建服务端节点
    server = GroundingDinoGraspServer()
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info("👋 用户中断，服务端退出")
    finally:
        # 清理资源（关闭可视化窗口等）
        cv2.destroyAllWindows()
        server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()