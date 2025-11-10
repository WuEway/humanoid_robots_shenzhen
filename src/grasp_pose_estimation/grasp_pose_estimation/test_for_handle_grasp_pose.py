import numpy as np
import open3d as o3d

import sys
import os
# ✅ 导入你的抓取估计器
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grasp_pose_estimation.handle_grasp_pose_estimation import HandleGraspEstimator  

# 1. 读取 .pcd 文件
pcd_path = os.path.join(os.path.dirname(__file__), '../../../takeout_bag.pcd')
data = o3d.io.read_point_cloud(pcd_path)
# print(f"读取 {pcd_path}，包含字段：{data.files}")

# 2. 恢复点云
points = np.asarray(data.points)
colors = np.asarray(data.colors)        # 颜色范围【0， 1】
print(f"点数: {points.shape[0]}")
print(f"随机颜色点: {points[100]}, 颜色: {colors[100]}")

# 3. 转换为 Open3D 点云 (可视化用)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"点云已加载，包含 {len(pcd.points)} 个点。")

# ✅ 可选：显示点云，确认读取无误
o3d.visualization.draw_geometries([pcd], window_name="Loaded current_pcd")

# 4. 初始化抓取估计器
grasp_estimator = HandleGraspEstimator(
            voxel_size=0.005,              # 提手点云内部处理的体素大小
            dbscan_eps=0.02,               # 提手聚类Eps
            dbscan_min_points=30,
            hsv_v_max=0.2,                # 黑色/深棕色的亮度阈值
            hsv_s_max=0.9,                 # 黑色/深棕色的饱和度阈值
            u_shape_min_points=50,         # U形簇最小点数
            u_shape_central_ratio=0.4,     # U形检测中心区域比例
            u_shape_hollow_ratio=0.15,     # U形空心比例
            grasp_bottom_height=0.03,      # 抓取点计算高度 (z_min + 0.03m)
            visualize=True
        )

# 5. 调用抓取位姿计算
acc_points = np.asarray(pcd.points)
acc_colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8) # 颜色转回 0-255
# 计算耗时
import time
t_start = time.time()
result = grasp_estimator.calculate_grasp_pose(acc_points, acc_colors)
t_end = time.time()
print(f"抓取位姿计算耗时: {t_end - t_start:.2f} 秒")

# 6. 输出结果
print("抓取位姿计算结果：")
print(result)
