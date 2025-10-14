"""
Grasp Pose Estimator
从给定的点云中计算抓取位姿
"""
import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from typing import Optional, Tuple

class GraspPoseEstimator:
    """从点云中估计抓取位姿的类"""

    def __init__(self,
                 voxel_size: float = 0.002,
                 stat_outlier_neighbors: int = 30,
                 stat_outlier_std_ratio: float = 0.8,
                 dbscan_eps: float = 0.03,
                 dbscan_min_points: int = 50,
                 final_stat_outlier_std_ratio: float = 0.5,
                 final_radius_outlier_nb_points: int = 20,
                 final_radius_outlier_radius: float = 0.02,
                 edge_candidate_tolerance: float = 0.020,
                 ransac_residual_threshold: float = 0.010,
                 black_threshold: float = 0.2,
                 visualize: bool = False):
        """
        初始化抓取位姿估计器

        Args:
            voxel_size: 体素下采样大小
            stat_outlier_neighbors: 统计离群点移除的邻居点数
            stat_outlier_std_ratio: 统计离群点移除的标准差倍数
            dbscan_eps: DBSCAN聚类的邻域半径
            dbscan_min_points: DBSCAN聚类的核心点最小邻居数
            final_stat_outlier_std_ratio: 对最大聚类进行最终清理的离群点标准差倍数
            final_radius_outlier_nb_points: 对最大聚类进行最终清理的半径离群点邻居数
            final_radius_outlier_radius: 对最大聚类进行最终清理的半径离群点半径
            edge_candidate_tolerance: 筛选边缘候选点的深度容差
            ransac_residual_threshold: RANSAC线段拟合的内点距离阈值
            black_threshold: 颜色阈值，用于滤除黑色点
            visualize: 是否可视化预处理前后的点云
        """
        self.voxel_size = voxel_size
        self.stat_outlier_neighbors = stat_outlier_neighbors
        self.stat_outlier_std_ratio = stat_outlier_std_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.final_stat_outlier_std_ratio = final_stat_outlier_std_ratio
        self.final_radius_outlier_nb_points = final_radius_outlier_nb_points
        self.final_radius_outlier_radius = final_radius_outlier_radius
        self.edge_candidate_tolerance = edge_candidate_tolerance
        self.ransac_residual_threshold = ransac_residual_threshold
        self.black_threshold = black_threshold
        self.visualize = visualize

    def calculate_grasp_pose(self, pcd_points: np.ndarray, pcd_colors: np.ndarray) -> Optional[Tuple[Point, Quaternion]]:
        """
        从输入的点和颜色计算抓取位姿（中心点和方向）

        Args:
            pcd_points: (N, 3) 的点云坐标数组
            pcd_colors: (N, 3) 的点云颜色数组 (RGB, 0-255)

        Returns:
            一个元组 (grasp_point, grasp_orientation)，如果失败则返回 None
        """
        if pcd_points.shape[0] < self.dbscan_min_points:
            print("⚠️  点云数量过少，跳过抓取计算")
            return None

        # 1. 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors / 255.0)

        # 2. 点云预处理
        pcd_processed = self._preprocess_point_cloud(pcd)


        if pcd_processed is None or len(pcd_processed.points) < 2:
            print("⚠️  预处理后点云过少，无法计算抓取位姿")
            return None

        # 3. 使用 RANSAC 在 3D 上拟合直线（合并边缘候选筛选）并得到抓取位姿
        fit_res = self._fit_line_and_pose_ransac(pcd_processed)
        if fit_res is None:
            print("⚠️  未能拟合出抓取线段")
            return None
        grasp_point, grasp_orientation, line_points = fit_res
        
        # 可选：可视化预处理结果（原始对比 + 线段）
        if self.visualize:
            self._visualize_preprocessing(pcd_processed, line_points)
        
        if self.visualize:
            # 在处理后的点云上可视化抓取点与姿态
            try:
                self._visualize_grasp_on_processed(
                    pcd_processed=pcd_processed,
                    grasp_point=grasp_point,
                    grasp_orientation=grasp_orientation,
                    line_points=line_points,
                )
            except Exception as e:
                print(f"⚠️  可视化抓取姿态时出错: {e}")

        print(f"✅ 成功计算抓取位姿: Point({grasp_point.x:.3f}, {grasp_point.y:.3f}, {grasp_point.z:.3f})")
        return grasp_point, grasp_orientation

    def _preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """对点云进行滤波、聚类等预处理"""
        # 体素下采样
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 统计离群点移除
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.stat_outlier_neighbors, std_ratio=self.stat_outlier_std_ratio)

        # 创建一个布尔掩码来识别非黑色的点
        colors_rgb = np.asarray(pcd.colors)
        # 提取 R, G, B 通道
        r = colors_rgb[:, 0]
        g = colors_rgb[:, 1]
        b = colors_rgb[:, 2]
        non_black_mask = (r > self.black_threshold) | (g > self.black_threshold) | (b > self.black_threshold)
        pcd = pcd.select_by_index(np.where(non_black_mask)[0])
        
        # DBSCAN聚类，保留最大聚类
        labels = np.array(pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))
        if labels.max() < 0:
            print("⚠️  DBSCAN未能找到任何有效聚类")
            return None
        
        largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
        pcd_final = pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])

        # 对最大聚类再次进行严格滤波
        pcd_final, _ = pcd_final.remove_statistical_outlier(nb_neighbors=self.stat_outlier_neighbors, std_ratio=self.final_stat_outlier_std_ratio)
        pcd_final, _ = pcd_final.remove_radius_outlier(nb_points=self.final_radius_outlier_nb_points, radius=self.final_radius_outlier_radius)
        
        return pcd_final

    def _visualize_preprocessing(self, original_pcd: o3d.geometry.PointCloud, line_pcd: Optional[np.ndarray]):
        """
        可视化预处理前后的点云对比

        Args:
            original_pcd: 原始点云
            line_pcd: 提取的线段点云
        """
        print("🎨 显示预处理前后点云对比...")
        
        # 创建一个副本并将其染成灰色以作对比
        original_pcd_copy = o3d.geometry.PointCloud(original_pcd)
        original_pcd_copy.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色

        geometries = [original_pcd_copy]
        if line_pcd is not None and len(line_pcd) > 0:
            line_pcd_copy = o3d.geometry.PointCloud()
            line_pcd_copy.points = o3d.utility.Vector3dVector(line_pcd)
            line_pcd_copy.paint_uniform_color([1.0, 0.0, 0.0]) # 红色
            geometries.append(line_pcd_copy)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Preprocessing: Original (Grey) vs Processed (Color)",
            width=1024,
            height=768
        )

    def _visualize_grasp_on_processed(
        self,
        pcd_processed: o3d.geometry.PointCloud,
        grasp_point: Point,
        grasp_orientation: Quaternion,
        line_points: Optional[np.ndarray] = None,
        axis_size: float = 0.05,
    ) -> None:
        """
        在处理后的点云上叠加抓取中心与姿态坐标轴进行可视化。

        Args:
            pcd_processed: 预处理后的点云
            grasp_point: 抓取中心点（ROS Point）
            grasp_orientation: 抓取姿态四元数（ROS Quaternion，xyzw）
            line_points: 可选，用于展示拟合的线段点
            axis_size: 可视化坐标轴的尺寸（米）
        """
        # 复制点云，避免修改原对象的颜色
        pcd_vis = o3d.geometry.PointCloud(pcd_processed)

        # 将处理后的点云整体染为浅灰，方便突出坐标轴与线段
        try:
            pcd_vis.paint_uniform_color([0.65, 0.65, 0.65])
        except Exception:
            pass

        # 构建 grasp 姿态的旋转矩阵
        quat_xyzw = np.array([
            grasp_orientation.x,
            grasp_orientation.y,
            grasp_orientation.z,
            grasp_orientation.w,
        ])
        R_mat = R.from_quat(quat_xyzw).as_matrix()

        # 构建位姿变换矩阵，将坐标系放在抓取中心
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = np.array([grasp_point.x, grasp_point.y, grasp_point.z])

        # 使用 Open3D 的坐标系网格展示抓取姿态
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        frame.transform(T)

        geometries = [pcd_vis, frame]

        # 如果有拟合线段点，叠加显示为红色点云
        if line_points is not None and len(line_points) > 0:
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line_points)
            line_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            geometries.append(line_pcd)

        # 在抓取中心放一个小球标记
        try:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(axis_size * 0.2, 1e-3))
            sphere.paint_uniform_color([1.0, 0.0, 1.0])  # 品红色，容易辨认
            sphere.compute_vertex_normals()
            T_s = np.eye(4)
            T_s[:3, 3] = T[:3, 3]
            sphere.transform(T_s)
            geometries.append(sphere)
        except Exception:
            pass

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Processed PCD with Grasp Pose",
            width=1024,
            height=768,
        )

    def _fit_line_and_pose_ransac(self, pcd: o3d.geometry.PointCloud) -> Optional[Tuple[Point, Quaternion, np.ndarray]]:
        """
        合并边缘候选筛选与 3D 直线拟合：
        1) 先按 z 轴靠近边缘的策略筛选候选点；
        2) 用 RANSAC 多输出回归在 3D 空间拟合一条直线；
        3) 由直线方向和内点中心计算抓取位姿。

        Returns:
            (grasp_point_msg, grasp_orientation_msg, inlier_points) 或 None
        """
        P = np.asarray(pcd.points)
        if P.shape[0] < 2:
            return None

        # 1) 候选点：边缘筛选
        depth_axis_index = 2
        z_max = np.max(P[:, depth_axis_index])
        candidate_mask = P[:, depth_axis_index] > (z_max - self.edge_candidate_tolerance)
        P_cand = P[candidate_mask]
        if P_cand.shape[0] < 2:
            # 如果候选点过少，退化为全部点
            P_cand = P
        N = P_cand.shape[0]

        # 2) 选择方差最大的轴作为自变量，做多输出线性回归 Y=[other two axes]
        var3 = np.var(P_cand, axis=0)
        main_idx = int(np.argmax(var3))
        other = [i for i in [0, 1, 2] if i != main_idx]
        X = P_cand[:, main_idx].reshape(-1, 1)
        Y = P_cand[:, other]  # (N,2)

        # RANSAC 拟合
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self.ransac_residual_threshold,
            min_samples=max(2, min(50, int(0.2 * N))),
            random_state=0,
        )
        try:
            ransac.fit(X, Y)
        except ValueError:
            return None

        inlier_mask = getattr(ransac, "inlier_mask_", None)
        if inlier_mask is None or np.count_nonzero(inlier_mask) < 2:
            return None

        # 方向向量：在 main_idx 维度的斜率为 1，其它两维为回归系数
        coef = np.array(ransac.estimator_.coef_).reshape(2,)
        direction = np.zeros(3, dtype=float)
        direction[main_idx] = 1.0
        direction[other[0]] = coef[0]
        direction[other[1]] = coef[1]
        # 归一化
        nrm = np.linalg.norm(direction)
        if nrm < 1e-12:
            return None
        direction /= nrm

        # 选择一个线上的代表点：选取内点 X 的中位数对应的位置
        X_in = X[inlier_mask]
        x0 = float(np.median(X_in))
        yz0 = ransac.predict([[x0]])[0]  # (2,)
        p0 = np.zeros(3, dtype=float)
        p0[main_idx] = x0
        p0[other[0]] = yz0[0]
        p0[other[1]] = yz0[1]

        # 以Y轴为抓取方向，从上往下抓取
        y_axis = direction
        z_axis_world = np.array([0.0, 0.0, -1.0])
        x_axis = np.cross(y_axis, z_axis_world)
        if np.linalg.norm(x_axis) < 1e-6:
            z_axis_world = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(y_axis, z_axis_world)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        R_mat = np.array([x_axis, y_axis, z_axis]).T
        quat = R.from_matrix(R_mat).as_quat()  # xyzw

        grasp_point_msg = Point(x=p0[0], y=p0[1], z=p0[2])
        grasp_orientation_msg = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # 返回用于可视化的内点集合
        inlier_points = P_cand[inlier_mask]
        return grasp_point_msg, grasp_orientation_msg, inlier_points
