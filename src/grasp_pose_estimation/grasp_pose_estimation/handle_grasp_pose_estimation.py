"""
Handle Grasp Pose Estimator
从给定的点云中检测U形提手并计算抓取位姿

实现逻辑:
1. 颜色分割 (HSV): 提取黑色/深色点云。
2. 空间聚类 (DBSCAN): 将黑点分成不同簇。
3. U形检测 (Hollow Check): 遍历所有簇，通过检查其2D投影的“中心区域”是否为空来识别U形。
4. 目标选择: 选择点数最多的那个U形簇。
5. 位姿计算:
    - 抓取点: 目标簇 Z 坐标最低的 3cm 范围内的点云均值。
    - Z轴: 竖直向下 (0, 0, -1)。
    - Y轴: 目标簇第一主成分(PCA)的水平(XY)投影。
    - X轴: 右手定则 (Y x Z)。
"""
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from typing import Optional, Tuple
import matplotlib.colors
import matplotlib.pyplot as plt

class HandleGraspEstimator:
    """从点云中估计U形提手抓取位姿的类"""

    def __init__(self,
                 voxel_size: float = 0.005,
                 dbscan_eps: float = 0.02,
                 dbscan_min_points: int = 30,
                 hsv_v_max: float = 0.3,
                 hsv_s_max: float = 0.5,
                 u_shape_min_points: int = 100,
                 u_shape_central_ratio: float = 0.4,
                 u_shape_hollow_ratio: float = 0.1,
                 grasp_bottom_height: float = 0.03,
                 visualize: bool = False):
        """
        初始化U形提手抓取位姿估计器

        Args:
            voxel_size: 提手点云下采样大小
            dbscan_eps: DBSCAN聚类的邻域半径
            dbscan_min_points: DBSCAN聚类的核心点最小邻居数
            hsv_v_max: 黑色过滤的HSV亮度(V)最大值 (0-1)
            hsv_s_max: 黑色过滤的HSV饱和度(S)最大值 (0-1)
            u_shape_min_points: 被视为U形簇的最小点数
            u_shape_central_ratio: U形检测中“中心区域”的边长比例
            u_shape_hollow_ratio: U形检测中“中心区域”点数占总点数的最大比例
            grasp_bottom_height: 从z_min向上计算抓取点的范围 (米)
            visualize: 是否可视化中间步骤
        """
        self.voxel_size = voxel_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.hsv_v_max = hsv_v_max
        self.hsv_s_max = hsv_s_max
        self.u_shape_min_points = u_shape_min_points
        self.u_shape_central_ratio = u_shape_central_ratio
        self.u_shape_hollow_ratio = u_shape_hollow_ratio
        self.grasp_bottom_height = grasp_bottom_height
        self.visualize = visualize

    def calculate_grasp_pose(self, pcd_points: np.ndarray, pcd_colors: np.ndarray) -> Optional[Tuple[Point, Quaternion]]:
        """
        从输入的点和颜色计算抓取位姿（中心点和方向）

        Args:
            pcd_points: (N, 3) 的点云坐标数组 (应在 base_link 坐标系下)
            pcd_colors: (N, 3) 的点云颜色数组 (RGB, 0-255)

        Returns:
            一个元组 (grasp_point, grasp_orientation)，如果失败则返回 None
        """
        if pcd_points.shape[0] < self.dbscan_min_points:
            print("⚠️ [Handle] 累积点云数量过少，跳过抓取计算")
            return None

        # 1. 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors / 255.0)

        # 2. 预处理
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 3. 颜色滤波 (提取黑色/深色点)
        black_pcd = self._filter_black_points(pcd)
        if not black_pcd.has_points() or len(black_pcd.points) < self.dbscan_min_points:
            print("⚠️ [Handle] 未能通过颜色滤波找到足够的提手点")
            return None

        # 4. 空间聚类 (DBSCAN)
        labels = np.array(black_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))
        unique_labels = set(labels)
        
        if self.visualize:
            # 方案1: 只显示有效聚类，完全忽略噪声
            valid_mask = labels >= 0  # 只保留非噪声点
            
            # 创建只包含有效聚类的点云
            valid_labels = labels[valid_mask]
            valid_pcd = black_pcd.select_by_index(np.where(valid_mask)[0])

            # 为有效聚类分配颜色
            if len(valid_labels) > 0:
                # colors = plt.get_cmap("tab20")(valid_labels / (valid_labels.max() if valid_labels.max() > 0 else 1))
                # valid_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                # o3d.visualization.draw_geometries([valid_pcd], window_name="有效黑色聚类 (无噪声)")
                num_clusters = len(set(labels[valid_mask]))
                print(f"聚类结果: {num_clusters} 个有效簇")
                
                # 直接使用原始颜色可视化
                o3d.visualization.draw_geometries([valid_pcd], window_name=f"黑色聚类 ({num_clusters}",)
            else:
                print("⚠️ 没有找到有效聚类")


        # 5. U形检测 (Hollow Check)
        u_shape_clusters = [] # 存储 (pcd, point_count)
        for label in unique_labels:
            if label < 0:
                continue # 忽略噪声
            
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < self.u_shape_min_points:
                continue # 簇太小

            cluster_pcd = black_pcd.select_by_index(cluster_indices)
            
            # 检查是否为空心U形
            if self._is_u_shape(cluster_pcd):
                u_shape_clusters.append((cluster_pcd, len(cluster_pcd.points)))
                print(f"✅ [Handle] 发现U形簇: 标签 {label}, 点数 {len(cluster_pcd.points)}")
            else:
                print(f"❌ [Handle] 丢弃实心簇: 标签 {label}, 点数 {len(cluster_pcd.points)}")

        # 6. 目标提手选择
        if not u_shape_clusters:
            print("⚠️ [Handle] 未能找到任何U形提手簇")
            return None
        
        # 按点数排序，选择点数最多的
        u_shape_clusters.sort(key=lambda x: x[1], reverse=True)
        target_handle_pcd = u_shape_clusters[0][0]
        print(f"🎯 [Handle] 选定点数最多的U形簇 (共 {u_shape_clusters[0][1]} 点)")

        # 7. 计算抓取位姿
        grasp_pose_result = self._calculate_pose_from_handle(target_handle_pcd)
        
        if grasp_pose_result is None:
            print("⚠️ [Handle] 计算最终位姿失败")
            return None
            
        grasp_point, grasp_orientation, grasp_points_for_vis = grasp_pose_result

        # 8. 可视化
        if self.visualize:
            self._visualize_grasp_on_handle(
                target_handle_pcd,
                grasp_point,
                grasp_orientation,
                grasp_points_for_vis
            )

        print(f"✅ [Handle] 成功计算抓取位姿: Point({grasp_point.x:.3f}, {grasp_point.y:.3f}, {grasp_point.z:.3f})")
        return grasp_point, grasp_orientation

    def _filter_black_points(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """使用HSV空间过滤黑色/深色点"""
        colors_rgb = np.asarray(pcd.colors)
        if colors_rgb.shape[0] == 0:
            return o3d.geometry.PointCloud()
            
        colors_hsv = matplotlib.colors.rgb_to_hsv(colors_rgb)
        
        # 提取 H, S, V
        # H = colors_hsv[:, 0]
        S = colors_hsv[:, 1]
        V = colors_hsv[:, 2]
        
        # 黑色/深色的V值和S值都较低
        mask = (V < self.hsv_v_max) & (S < self.hsv_s_max)
        
        black_pcd = pcd.select_by_index(np.where(mask)[0])
        return black_pcd

    def _is_u_shape(self, pcd: o3d.geometry.PointCloud) -> bool:
        """
        检查点云簇是否为“空心”U形。
        方法：将其投影到XY平面，检查其中心区域的点密度。
        """
        P = np.asarray(pcd.points)
        # P_2d = P[:, :2] # 投影到XY平面

        # 使用PCA计算投影平面
        try:
            # 计算点云的均值和协方差矩阵
            mean = np.mean(P, axis=0)
            P_centered = P - mean
            cov = np.cov(P_centered.T)
            
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 按特征值从大到小排序
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 获取前两个主成分（最大方差的两个方向）
            pc1 = eigenvectors[:, 0]  # 第一主成分
            pc2 = eigenvectors[:, 1]  # 第二主成分
            
            # 检查前两个特征值是否足够大（避免退化情况）
            if eigenvalues[0] < 1e-6 or eigenvalues[1] < 1e-6:
                return False  # 点云过于扁平或呈线状
            
            # 将点云投影到前两个主成分张成的平面
            # P_2d[i] = [P_centered[i] · pc1, P_centered[i] · pc2]
            P_2d = np.column_stack([
                np.dot(P_centered, pc1),
                np.dot(P_centered, pc2)
            ])
            
        except (ValueError, np.linalg.LinAlgError):
            print("⚠️ [Handle] PCA计算失败，无法投影点云到2D平面")
            return False  # PCA计算失败

        try:
            # 计算2D AABB (轴对齐边界框)
            x_min, y_min = np.min(P_2d, axis=0)
            x_max, y_max = np.max(P_2d, axis=0)
        except ValueError:
            return False # 点云为空

        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range < 1e-2 or y_range < 1e-2:
            return False # 是条线，不是U形

        # 定义中心区域 (例如，一个缩小40%的框)
        ratio = self.u_shape_central_ratio
        cx_min = x_min + x_range * (0.5 - ratio / 2)
        cx_max = x_min + x_range * (0.5 + ratio / 2)
        cy_min = y_min + y_range * (0.5 - ratio / 2)
        cy_max = y_min + y_range * (0.5 + ratio / 2)

        # 统计落在中心区域的点
        mask_x = (P_2d[:, 0] >= cx_min) & (P_2d[:, 0] <= cx_max)
        mask_y = (P_2d[:, 1] >= cy_min) & (P_2d[:, 1] <= cy_max)
        
        count_central = np.count_nonzero(mask_x & mask_y)
        total_points = P.shape[0]
        
        hollow_ratio = count_central / total_points

        if self.visualize:
            self._visualize_pca_projection(
                pcd, P_2d,
                x_min, x_max, y_min, y_max,
                cx_min, cx_max, cy_min, cy_max,
                hollow_ratio < self.u_shape_hollow_ratio
            )

        # 如果中心区域的点数比例低于阈值，则认为是“空心”U形
        return hollow_ratio < self.u_shape_hollow_ratio
    
    def _visualize_pca_projection(self, pcd: o3d.geometry.PointCloud, P_2d: np.ndarray, 
                                x_min: float, x_max: float, y_min: float, y_max: float,
                                cx_min: float, cx_max: float, cy_min: float, cy_max: float,
                                is_u_shape: bool) -> None:
        """
        可视化PCA投影后的2D点云和中心检测框
        
        Args:
            pcd: 原始3D点云
            P_2d: 投影后的2D点云 (N, 2)
            x_min, x_max, y_min, y_max: 2D包围盒边界
            cx_min, cx_max, cy_min, cy_max: 中心区域边界
            is_u_shape: 是否检测为U形
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- 左图: 2D投影点云 ---
        ax1.scatter(P_2d[:, 0], P_2d[:, 1], c='blue', s=5, alpha=0.6, label='projected cloud')
        
        # 绘制外部包围盒
        bbox_x = [x_min, x_max, x_max, x_min, x_min]
        bbox_y = [y_min, y_min, y_max, y_max, y_min]
        ax1.plot(bbox_x, bbox_y, 'g-', linewidth=2, label='bounding box')
        
        # 绘制中心检测区域
        central_x = [cx_min, cx_max, cx_max, cx_min, cx_min]
        central_y = [cy_min, cy_min, cy_max, cy_max, cy_min]
        ax1.plot(central_x, central_y, 'r--', linewidth=2, label='central region')
        
        # 标记中心区域的点
        mask_x = (P_2d[:, 0] >= cx_min) & (P_2d[:, 0] <= cx_max)
        mask_y = (P_2d[:, 1] >= cy_min) & (P_2d[:, 1] <= cy_max)
        central_points = P_2d[mask_x & mask_y]
        
        if central_points.shape[0] > 0:
            ax1.scatter(central_points[:, 0], central_points[:, 1], 
                    c='red', s=10, alpha=0.8, label=f'central points ({len(central_points)})')

        ax1.set_xlabel('PC1 (First Principal Component)')
        ax1.set_ylabel('PC2 (Second Principal Component)')
        ax1.set_title(f'PCA Projection to 2D Plane\n{"[YES] U-Shape" if is_u_shape else "[NO] Solid"}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # --- 右图: 3D原始点云俯视图 ---
        P = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        ax2.scatter(P[:, 0], P[:, 1], c=colors if colors.size > 0 else 'gray', 
                s=5, alpha=0.6, label='Original Point Cloud (XY View)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('3D Point Cloud Top View (XY Plane)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

    def _calculate_pose_from_handle(self, handle_pcd: o3d.geometry.PointCloud) -> Optional[Tuple[Point, Quaternion, np.ndarray]]:
        """从U形提手点云计算抓取位姿"""
        P = np.asarray(handle_pcd.points)
        if P.shape[0] < 3:
            return None

        # --- 1. 计算抓取点 (Position) ---
        z_min = np.min(P[:, 2])
        mask = (P[:, 2] >= z_min) & (P[:, 2] <= (z_min + self.grasp_bottom_height))
        grasp_points = P[mask]
        
        if grasp_points.shape[0] == 0:
            # 如果在3cm范围内没有点，则使用z_min的那个点
            grasp_points = P[np.argmin(P[:, 2]).reshape(1, -1)]
            
        # 抓取点 = 底部点云的均值
        p0 = np.mean(grasp_points, axis=0)
        grasp_point_msg = Point(x=p0[0], y=p0[1], z=p0[2])

        # --- 2. 计算抓取姿态 (Orientation) ---
        
        # Z 轴: 竖直向下
        Z_grasp = np.array([0.0, 0.0, -1.0])

        # X 轴: 提手平面法向量在水平(XY)上的投影
        try:
            # PCA 找到提手的主平面
            mean, cov = handle_pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 第三主成分 (最小特征值对应的方向) = 平面法向量
            pc3_3d = eigenvectors[:, 0]  # 最小特征值对应的特征向量
            
        except Exception:
            # 如果PCA失败 (例如点太少)，使用默认X轴
            pc3_3d = np.array([1.0, 0.0, 0.0])
            
        # 将平面法向量投影到XY平面
        v_x = np.array([pc3_3d[0], pc3_3d[1], 0.0])
        norm_x = np.linalg.norm(v_x)
        
        if norm_x < 1e-6:
            # 如果法向量是竖直的 (平面是水平的)
            # 退化情况：使用第二主成分投影
            pc2_3d = eigenvectors[:, 1]
            v_x = np.array([pc2_3d[0], pc2_3d[1], 0.0])
            norm_x = np.linalg.norm(v_x)
            
            if norm_x < 1e-6:
                # 还是失败，用默认 X=(1,0,0)
                v_x = np.array([1.0, 0.0, 0.0])
                norm_x = 1.0

        X_grasp = v_x / norm_x
        
        # 确保X轴方向一致性 (可选，但推荐)
        # 假设我们希望 X 轴大致指向 X 轴正方向
        if X_grasp[0] < 0:
            X_grasp = -X_grasp
            
        # Y 轴: 右手定则 Y = Z × X
        Y_grasp = np.cross(Z_grasp, X_grasp)
        Y_grasp /= np.linalg.norm(Y_grasp)  # 归一化
        
        # 重新计算 X 轴确保严格正交
        X_grasp = np.cross(Y_grasp, Z_grasp)
        X_grasp /= np.linalg.norm(X_grasp)

        # 转换为旋转矩阵和四元数
        R_mat = np.array([X_grasp, Y_grasp, Z_grasp]).T
        quat = R.from_matrix(R_mat).as_quat()  # xyzw

        grasp_orientation_msg = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        return grasp_point_msg, grasp_orientation_msg, grasp_points

    def _visualize_grasp_on_handle(
        self,
        handle_pcd: o3d.geometry.PointCloud,
        grasp_point: Point,
        grasp_orientation: Quaternion,
        grasp_points_for_vis: np.ndarray,
        axis_size: float = 0.05,
    ) -> None:
        """在提手点云上可视化抓取位姿"""
        
        print("🎨 [Handle] 显示最终抓取位姿...")
        
        # 提手点云染成灰色
        pcd_vis = o3d.geometry.PointCloud(handle_pcd)
        pcd_vis.paint_uniform_color([0.5, 0.5, 0.5])

        # 抓取区域点云染成红色
        grasp_pcd = o3d.geometry.PointCloud()
        grasp_pcd.points = o3d.utility.Vector3dVector(grasp_points_for_vis)
        grasp_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # 构建 grasp 姿态的旋转矩阵
        quat_xyzw = np.array([
            grasp_orientation.x,
            grasp_orientation.y,
            grasp_orientation.z,
            grasp_orientation.w,
        ])
        R_mat = R.from_quat(quat_xyzw).as_matrix()

        # 构建位姿变换矩阵
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = np.array([grasp_point.x, grasp_point.y, grasp_point.z])

        # 坐标系
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        frame.transform(T)
        
        # 世界坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 2)

        geometries = [pcd_vis, grasp_pcd, frame, world_frame]
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Handle Grasp Pose",
            width=1024,
            height=768,
        )