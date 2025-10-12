"""
GroundingDino处理器的详细实现
包含目标检测和点云提取功能
集成真实的GroundingDino+SAM模型
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

from realsense_system import RGBDProcessor, RGBDData




class AdvancedGroundingDinoProcessor(RGBDProcessor):
    """高级GroundingDino处理器 - 集成真实的GroundingDino+SAM模型"""
    
    def __init__(self, 
                 grounding_dino_config_path: str = "/home/yiwei/code_from_web/cv_algorithms/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint_path: str = "/home/yiwei/data_repo/Grounded-SAM/groundingdino_swint_ogc.pth",
                 sam_encoder_version: str = "vit_h",
                 sam_checkpoint_path: str = "/home/yiwei/data_repo/Grounded-SAM/sam_vit_h_4b8939.pth",
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
        
        # 点云转换器
        self.pc = rs.pointcloud()
        self.points = rs.points()
        
        # 模型路径
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounding_dino_checkpoint_path = grounding_dino_checkpoint_path
        self.sam_encoder_version = sam_encoder_version
        self.sam_checkpoint_path = sam_checkpoint_path
        
        # 初始化模型
        self.grounding_dino_model = None
        self.sam_predictor = None
        
        # ICP配准相关属性 - 按物体类别分别维护
        self.accumulated_pcds = {}   # 每个物体类别维护独立的累积点云 {label: pointcloud}
        self.frame_count = 0         # 帧计数器
        
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
        
    def process(self, data: RGBDData, text_prompt: str = "object") -> Dict[str, Any]:
        """
        处理RGBD数据，检测目标并提取点云
        
        Args:
            data: RGBD数据
            text_prompt: 检测目标的文本描述
            
        Returns:
            包含检测结果和点云的字典
        """
        if not data.is_valid():
            return {"success": False, "error": "Invalid RGBD data"}
            
        try:
            # 更新帧计数器
            self.frame_count += 1
            # 1. 目标检测
            detections = self._detect_objects(data.color_image, text_prompt)
            
            # 2. 为每个检测结果提取点云
            point_clouds = []
            for detection in detections:
                point_cloud_data = self._extract_point_cloud(data, detection)
                if point_cloud_data is not None:
                    point_clouds.append({
                        "detection": detection,
                        "point_cloud": point_cloud_data
                    })
            
            # 3. 可视化结果
            result_image = self._visualize_detections(data.color_image, detections)
            
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
        
        # 按类别分组
        class_groups = {}
        for detection in detections:
            class_id = detection["class_id"]
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(detection)
        
        # 每个类别保留置信度最高的一个
        filtered_detections = []
        for class_id, group in class_groups.items():
            # 按置信度排序，取最高的
            best_detection = max(group, key=lambda x: x["confidence"])
            filtered_detections.append(best_detection)
            
            print(f"📦 类别 '{best_detection['label']}': 保留置信度最高的检测 ({best_detection['confidence']:.3f})")
        
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
    

    
    def _extract_point_cloud(self, data: RGBDData, detection: Dict) -> Optional[Dict]:
        """
        根据检测结果提取目标物体的点云（包含颜色信息）
        
        Args:
            data: RGBD数据
            detection: 检测结果
            
        Returns:
            包含点云和颜色信息的字典 {"points": (N, 3), "colors": (N, 3)} 或 None
        """
        try:
            if data.depth_frame is None:
                return None
                
            # 必须使用SAM生成的mask，确保精确分割
            mask = detection.get("mask")
            if mask is None:
                print("⚠️  警告: 检测结果中没有mask，跳过点云提取")
                return None
            
            # 生成点云
            self.pc.map_to(data.color_frame)
            self.points = self.pc.calculate(data.depth_frame)
            
            # 获取顶点坐标和颜色
            vertices = np.asanyarray(self.points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex_coords = np.asanyarray(self.points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            
            # 获取RGB图像数据
            color_image = np.asanyarray(data.color_frame.get_data())
            h, w = color_image.shape[:2]
            
            # 应用掩码过滤点云
            mask_flat = mask.flatten()
            valid_indices = np.where(mask_flat > 0)[0]
            
            if len(valid_indices) == 0:
                return None
                
            # 提取目标点云和对应的纹理坐标
            target_points = vertices[valid_indices]
            target_tex_coords = tex_coords[valid_indices]
            
            # 过滤无效点（z=0的点）
            valid_mask = target_points[:, 2] > 0
            valid_points = target_points[valid_mask]
            valid_tex_coords = target_tex_coords[valid_mask]
            
            if len(valid_points) == 0:
                return None
            
            # 从RGB图像中提取对应的颜色
            colors = []
            for tex_coord in valid_tex_coords:
                # 纹理坐标转换为像素坐标
                x = int(tex_coord[0] * w)
                y = int(tex_coord[1] * h)
                
                # 确保坐标在图像范围内
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                
                # 获取RGB颜色 (注意OpenCV是BGR格式)
                b, g, r = color_image[y, x]
                colors.append([r, g, b])  # 转换为RGB格式
            
            colors = np.array(colors, dtype=np.uint8)
            
            return {
                "points": valid_points,
                "colors": colors
            }
            
        except Exception as e:
            print(f"点云提取失败: {e}")
            return None
    
    def _visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """可视化检测结果"""
        result_image = image.copy()
        
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
    

    



class InteractiveGroundingDinoVisualizer(RGBDProcessor):
    """交互式GroundingDino可视化器"""
    
    def __init__(self, window_name: str = "GroundingDino Results"):
        self.window_name = window_name
        
        # 初始化处理器
        self.processor = AdvancedGroundingDinoProcessor()
        print("🤖 使用GroundingDino+SAM模型")
            
        self.current_prompt = "delivery box. pink takeout bag"  # 默认检测外卖袋
        self.frame_count = 0
        self.last_process_frame = 0
        self.last_results = None  # 缓存最近一次的检测结果
        
        # ICP配准相关 - 按物体类别分别维护
        self.accumulated_pcds = {}  # 每个物体类别维护独立的累积点云 {label: pointcloud}
        self.enable_icp = True  # 是否启用ICP配准
        
    def process(self, data: RGBDData) -> bool:
        """处理并显示检测结果"""
        if not data.is_valid():
            return True
            
        self.frame_count += 1
        
        # 每30帧处理一次（降低计算频率，真实模型计算较慢）
        if self.frame_count - self.last_process_frame >= 30:
            self.last_process_frame = self.frame_count
            print(f"\n=== 第 {self.frame_count} 帧 - 检测目标: '{self.current_prompt}' ===")
            
            # 执行检测
            self.last_results = self.processor.process(data, self.current_prompt)
            
            if self.last_results["success"]:
                # 显示结果
                result_image = self.last_results["result_image"]
                cv2.imshow(self.window_name, result_image)
                
                # 自动保存检测结果
                self._save_detection_results(data, self.last_results)
                
                # 打印检测信息
                detections = self.last_results["detections"]
                if detections:
                    print(f"✅ 检测到 {len(detections)} 个目标:")
                    for i, det in enumerate(detections):
                        print(f"  {i+1}. {det['label']}: {det['confidence']:.3f}")
                        
                        # 如果有点云，打印点云信息
                        point_clouds = self.last_results["point_clouds"]
                        if i < len(point_clouds) and point_clouds[i]["point_cloud"] is not None:
                            pc_size = len(point_clouds[i]["point_cloud"]["points"])
                            print(f" 点云大小: {pc_size} 个点")
                else:
                    print(f"❌ 未检测到目标: '{self.current_prompt}'")
            else:
                # 显示原图
                cv2.imshow(self.window_name, data.color_image)
                print(f"❌ 检测失败: {self.last_results.get('error', 'Unknown error')}")
                self.last_results = None
        # else:
        #     # 显示原图或上一次的结果
        #     cv2.imshow(self.window_name, data.color_image)

        
        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q或ESC退出
            return False
        elif key == ord('1'):  # 1检测外卖袋
            self.current_prompt = "delivery box. pink takeout bag"
            print(f"\n🔄 切换检测目标: {self.current_prompt}")
        elif key == ord('2'):  # 2检测包
            self.current_prompt = "bag"
            print(f"\n🔄 切换检测目标: {self.current_prompt}")
        elif key == ord('3'):  # 3检测盒子
            self.current_prompt = "box"
            print(f"\n🔄 切换检测目标: {self.current_prompt}")
        elif key == ord('i'):  # i开启/关闭ICP配准
            self.enable_icp = not self.enable_icp
            status = "开启" if self.enable_icp else "关闭"
            print(f"\n🔄 ICP配准功能已{status}")
            if not self.enable_icp:
                print("💡 禁用ICP后只保存原始点云")
            
        return True

    
    def _save_detection_results(self, data: RGBDData, results: Dict):
        """保存检测结果：点云（无颜色+有颜色）和可视化图像"""
        try:
            import os
            
            # 创建结果目录
            results_dir = "detection_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 为当前帧创建子目录
            frame_name = f"frame_{self.frame_count:04d}_{self.current_prompt.replace(' ', '_')}"
            frame_dir = os.path.join(results_dir, frame_name)
            os.makedirs(frame_dir, exist_ok=True)
            
            detections = results["detections"]
            point_clouds = results["point_clouds"]
            
            # 1. 保存RGB图像叠加SAM分割结果和DINO检测框
            result_path = os.path.join(frame_dir, "detection_overlay.jpg")
            cv2.imwrite(result_path, results["result_image"])
            
            # 1.1. 额外保存jpg到专门的图片文件夹，方便查看
            # 创建图片库目录
            gallery_dir = "detection_results/image_gallery"
            os.makedirs(gallery_dir, exist_ok=True)
            cv2.imwrite(os.path.join(gallery_dir, f"{frame_name}.jpg"), results["result_image"])

            # 2. 为每个检测目标保存点云和执行ICP配准
            for i, detection in enumerate(detections):
                obj_name = f"{detection['label'].replace(' ', '_')}_{i:02d}"
                
                # 保存对应的点云
                if i < len(point_clouds) and point_clouds[i]["point_cloud"] is not None:
                    pc_data = point_clouds[i]["point_cloud"]
                    
                    # 保存原始点云（有颜色的）
                    pc_path = os.path.join(frame_dir, f"{obj_name}_pointcloud.ply")
                    self.save_point_cloud(pc_data, pc_path)
                    
                    # 执行ICP配准并更新全局配准文件（按物体类别分别维护）
                    if self.enable_icp:
                        obj_label = detection['label']  # 获取物体类别标签
                        self._perform_incremental_icp_by_class(pc_data, obj_label, obj_name)
            
            print(f"💾 已保存检测结果到: {frame_dir}")
            
        except Exception as e:
            print(f"❌ 保存检测结果失败: {e}")

    def save_point_cloud(self, point_cloud_data, filepath: str) -> bool:
        """保存点云到文件（所有点云都包含颜色信息）"""
        try:
            points = point_cloud_data["points"]
            colors = point_cloud_data["colors"]
            
            # 保存带颜色的PLY文件
            base_name = filepath.rsplit('.', 1)[0]
            rgb_filepath = f"{base_name}_rgb.ply"
            self._save_colored_ply(points, colors, rgb_filepath)
            print(f"彩色点云已保存到: {rgb_filepath}")
            
            return True
            
        except Exception as e:
            print(f"保存点云失败: {e}")
            return False
    
    def _save_colored_ply(self, points: np.ndarray, colors: np.ndarray, filepath: str):
        """保存带颜色的点云为PLY格式"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

    def _perform_incremental_icp_by_class(self, pc_data, obj_label: str, obj_name: str):
        """
        执行按物体类别分组的增量式ICP配准
        每个物体类别维护独立的累积点云和配准历史
        
        Args:
            pc_data: 点云数据
            obj_label: 物体类别标签 (如 "delivery_box", "pink_takeout_bag")
            obj_name: 具体实例名称 (如 "delivery_box_00")
        """
        try:
            import open3d as o3d
            import numpy as np
            import os
            
            # 标准化标签名称（去除空格和特殊字符）
            clean_label = obj_label.replace(' ', '_').replace('.', '_').lower()
            
            # 为每个物体类别创建独立的全局配准文件路径
            global_registered_path = f"detection_results/global_registered_{clean_label}.ply"
            
            # 从点云数据中提取点坐标和颜色
            points = pc_data["points"]
            colors = pc_data["colors"]
            
            # 创建当前帧点云对象（目标点云）
            current_pcd = o3d.geometry.PointCloud()
            current_pcd.points = o3d.utility.Vector3dVector(points)
            colors_normalized = colors.astype(np.float64) / 255.0
            current_pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            
            # 检查点云质量
            if len(current_pcd.points) < 10:
                print(f"⚠️ {obj_label} 点云点数太少({len(current_pcd.points)})，跳过ICP配准")
                return
            
            # 如果这是该类别的第一帧，直接保存为该类别的累积历史点云
            if clean_label not in self.accumulated_pcds:
                self.accumulated_pcds[clean_label] = current_pcd
                o3d.io.write_point_cloud(global_registered_path, current_pcd)
                print(f"🔄 {obj_label} 首帧点云已保存: {obj_name} (点数: {len(current_pcd.points)})")
                return
            
            # 获取该类别的历史累积点云
            class_accumulated_pcd = self.accumulated_pcds[clean_label]
            
            print(f"🔄 {obj_label} ICP配准: 历史点云({len(class_accumulated_pcd.points)}点) -> 当前帧({len(current_pcd.points)}点)")
            
            # 预处理：估计法向量
            current_pcd.estimate_normals()
            class_accumulated_pcd.estimate_normals()
            
            # ICP配准参数
            threshold = 0.02  # 对应点距离阈值(米)
            
            try:
                # 先尝试Colored ICP（保持颜色一致性）
                reg_p2p = o3d.pipelines.registration.registration_colored_icp(
                    source=class_accumulated_pcd,     # 源点云：该类别的历史累积点云
                    target=current_pcd,               # 目标点云：当前帧点云
                    max_correspondence_distance=threshold,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                print(f"✅ {obj_label} 使用Colored ICP配准成功")
                
            except RuntimeError as e:
                if "No correspondences found" in str(e):
                    print(f"⚠️ {obj_label} Colored ICP找不到对应点，回退到Point-to-Plane ICP")
                    # 回退到point-to-plane ICP
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        source=class_accumulated_pcd,
                        target=current_pcd,
                        max_correspondence_distance=threshold,
                        init=np.eye(4),
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                    )
                    print(f"✅ {obj_label} 使用Point-to-Plane ICP配准成功")
                else:
                    raise e
            
            print(f"📊 {obj_label} ICP配准结果 - 适应度: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}m")
            
            # 检查配准质量
            if reg_p2p.fitness > 0.1:  # fitness > 0.1 表示配准较好
                # 将变换应用到该类别的历史点云，使其配准到当前帧坐标系
                aligned_history_pcd = class_accumulated_pcd.transform(reg_p2p.transformation)
                
                # 合并配准后的历史点云与当前帧点云
                merged_pcd = aligned_history_pcd + current_pcd
                
                # 下采样以控制点云大小和去除重复点
                voxel_size = 0.002  # 2mm体素大小
                merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
                
                # 更新该类别的累积历史点云
                self.accumulated_pcds[clean_label] = merged_pcd
                
                # 保存该类别的更新全局点云
                o3d.io.write_point_cloud(global_registered_path, merged_pcd)
                
                print(f"✅ {obj_label} ICP配准成功: {obj_name}")
                print(f"   历史点云已配准到当前帧坐标系")
                print(f"   合并后点云: {len(merged_pcd.points)} 个点")
                print(f"   全局点云文件已更新: {global_registered_path}")
                
                # 保存变换矩阵（用于调试）
                transform_dir = "detection_results"
                transform_path = os.path.join(transform_dir, f"transform_{clean_label}_{self.frame_count:04d}.txt")
                np.savetxt(transform_path, reg_p2p.transformation)
                
            else:
                print(f"⚠️ {obj_label} ICP配准质量较差(fitness={reg_p2p.fitness:.3f})，使用当前帧替换")
                # 配准失败时，使用当前帧替换该类别的累积点云（可能是新的场景或大幅变化）
                voxel_size = 0.002
                downsampled_current = current_pcd.voxel_down_sample(voxel_size)
                self.accumulated_pcds[clean_label] = downsampled_current
                o3d.io.write_point_cloud(global_registered_path, downsampled_current)
                print(f"   {obj_label} 累积点云已重置: {len(downsampled_current.points)} 个点")
                
        except Exception as e:
            print(f"❌ {obj_label} ICP配准失败: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    from realsense_system import BagFileReader, RealSenseSystem
    
    # 使用交互式GroundingDino可视化器
    bag_file_path = "/home/yiwei/my_project/HumanoidRobots_shenzhen/takeout_bag.bag"
    
    print("🤖 启动GroundingDino+SAM外卖袋检测系统")
    
    reader = BagFileReader(bag_file_path, repeat_playback=False)
    system = RealSenseSystem(reader)
    
    # 添加交互式可视化器
    visualizer = InteractiveGroundingDinoVisualizer()
    system.add_processor(visualizer)
    
    print("\n" + "="*60)
    print("🎯 GroundingDino + RealSense 交互式演示")
    print("="*60)
    print("📋 按键说明:")
    print("  1 - 检测外卖袋 (pink takeout bag with black handles) 🥡")
    print("  2 - 检测包 (bag) 👜") 
    print("  3 - 检测盒子 (box) 📦")
    print("  i - 开启/关闭ICP配准功能 🔄")
    print("  q/ESC - 退出 👋")
    print("="*60)
    print("ℹ️  检测频率: 每30帧处理一次")
    print("ℹ️  自动保存: 每次检测成功时自动保存结果")
    print("ℹ️  ICP配准: 默认开启，所有点云配准到最新坐标系")
    print("ℹ️  保存位置: detection_results/ 目录")
    print("ℹ️  运行命令: python3 grounding_dino_processor.py")
    print("="*60 + "\n")
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
