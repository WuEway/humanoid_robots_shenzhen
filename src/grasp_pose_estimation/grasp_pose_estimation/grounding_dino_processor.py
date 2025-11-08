import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import torch
import torchvision

# 模型相关导入（放在需要时再导入，避免初始化耗时）

class AdvancedGroundingDinoProcessor:
    """负责模型加载、目标检测、分割和点云提取的处理器"""
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
        
        # 设备配置
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型路径验证
        self._validate_path(grounding_dino_config_path, "GroundingDino配置文件")
        self._validate_path(grounding_dino_checkpoint_path, "GroundingDino权重文件")
        self._validate_path(sam_checkpoint_path, "SAM权重文件")
        
        # 模型路径
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounding_dino_checkpoint_path = grounding_dino_checkpoint_path
        self.sam_encoder_version = sam_encoder_version
        self.sam_checkpoint_path = sam_checkpoint_path

        # 模型初始化
        self.grounding_dino_model = None
        self.sam_predictor = None
        self._load_models()
        
        print(f"GroundingDino处理器初始化完成，使用设备: {self.device}")
    
    def _validate_path(self, path: str, desc: str):
        """验证文件路径是否有效"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc}不存在：{path}")
        if not os.path.isfile(path):
            raise IsADirectoryError(f"{desc}不是文件：{path}")

    def _load_models(self):
        """加载GroundingDino和SAM模型"""
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
        """处理图像，返回检测结果和点云"""
        if color_image is None or depth_image is None:
            return {"success": False, "error": "Invalid image data"}
            
        try:
            # 目标检测
            detections = self._detect_objects(color_image, text_prompt)
            
            # 提取点云
            point_clouds = []
            for detection in detections:
                point_cloud_data = self._extract_point_cloud(color_image, depth_image, detection, camera_intrinsics)
                if point_cloud_data is not None:
                    point_clouds.append({
                        "detection": detection,
                        "point_cloud": point_cloud_data
                    })
            
            return {
                "success": True,
                "detections": detections,
                "point_clouds": point_clouds,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_objects(self, image: np.ndarray, text_prompt: str) -> List[Dict]:
        """使用GroundingDino检测目标，SAM分割"""
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
            return self._pick_best_detection_per_class(all_detections)
            
        except Exception as e:
            print(f"检测过程出错: {e}")
            return []
    
    def _pick_best_detection_per_class(self, detections: List[Dict]) -> List[Dict]:
        """每类目标只保留置信度最高的检测结果"""
        if not detections:
            return detections
        
        class_groups = {}
        for detection in detections:
            label = detection["label"]
            class_groups[label] = class_groups.get(label, []) + [detection]
        
        return [max(group, key=lambda x: x["confidence"]) for group in class_groups.values()]

    def _segment_with_sam(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """使用SAM对检测框进行分割"""
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in boxes:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            result_masks.append(masks[np.argmax(scores)])
        return result_masks
    
    def _extract_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray, detection: Dict, camera_intrinsics: Optional[Dict[str, float]] = None) -> Optional[Dict]:
        """从掩码区域提取点云（相机坐标系）"""
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
                z = depth / 1000.0  # 假设深度图单位为mm
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
