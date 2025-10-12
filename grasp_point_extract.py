import open3d as o3d
import numpy as np

# 1. 加载你的点云
pcd = o3d.io.read_point_cloud("detection_results/frame_0150_delivery_box._pink_takeout_bag/pink_takeout_bag_00_pointcloud_rgb.ply")

# 2. 可选：轻微的体素下采样进一步平滑
pcd = pcd.voxel_down_sample(voxel_size=0.002)

# 3. 应用统计离群点移除
# nb_neighbors: 指定邻居点的数量
# std_ratio: 标准差的倍数。这个值越小，过滤越严格
pcd_denoised, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.8)

# 确保点云有颜色信息
if not pcd_denoised.has_colors():
    raise ValueError("点云没有颜色信息！")

# 将颜色转换为 numpy 数组 (0-1范围)
colors_rgb = np.asarray(pcd_denoised.colors)

# 提取 R, G, B 通道
r = colors_rgb[:, 0]
g = colors_rgb[:, 1]
b = colors_rgb[:, 2]

# 创建一个布尔掩码来识别非黑色的点
black_threshold = 0.2
non_black_mask = (r > black_threshold) | (g > black_threshold) | (b > black_threshold)

# 4. 根据掩码选择点
pcd_pink_bag = pcd_denoised.select_by_index(np.where(non_black_mask)[0])

# 6. 🔥 新增：DBSCAN聚类分析，只保留最大的点云团
print("开始聚类分析...")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd_pink_bag.cluster_dbscan(eps=0.03, min_points=50, print_progress=True))

max_label = labels.max()
print(f"发现 {max_label + 1} 个聚类")

if max_label < 0:
    print("⚠️ 没有找到任何有效聚类！")
    pcd_final = pcd_pink_bag
else:
    # 统计每个聚类的点数
    cluster_sizes = []
    for i in range(max_label + 1):
        cluster_size = np.sum(labels == i)
        cluster_sizes.append((i, cluster_size))
        print(f"聚类 {i}: {cluster_size} 个点")
    
    # 找到最大的聚类
    largest_cluster_idx = max(cluster_sizes, key=lambda x: x[1])[0]
    largest_cluster_size = max(cluster_sizes, key=lambda x: x[1])[1]
    
    print(f"✅ 保留最大聚类 {largest_cluster_idx}: {largest_cluster_size} 个点")
    
    # 只保留最大聚类的点
    largest_cluster_mask = labels == largest_cluster_idx
    pcd_final = pcd_pink_bag.select_by_index(np.where(largest_cluster_mask)[0])

    # 7. 🔥 对最大聚类再次应用严格的滤波
    print("对主要物体进行最终清理...")
    
    # 再次统计离群点移除
    pcd_final, ind = pcd_final.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.5)
    
    # 再次半径离群点移除
    pcd_final, ind = pcd_final.remove_radius_outlier(nb_points=20, radius=0.02)
    
    print(f"点云从 {len(pcd.points)} 个点减少到 {len(pcd_final.points)} 个点")

# # 7. 可视化最终结果
# print("可视化清理后的点云...")
# o3d.visualization.draw_geometries([pcd_final], 
#                                   window_name="只保留最大点云团",
#                                   width=1024, height=768)

# ------ 提取抓取边缘点 ------
# 假设 pcd_final 是已经去除了噪声和提手的点云
points_3d = np.asarray(pcd_final.points)

# 1. 找到最小深度值 z_min (假设Z是深度轴)
depth_axis_index = 2
z_min = np.min(points_3d[:, depth_axis_index])

# 2. 设置一个稍大的容差来获取候选点
# 单位需要和你的点云单位匹配，这里假设是米
edge_candidate_tolerance = 0.015 # 1.5 cm

# 3. 筛选出候选点
candidate_mask = points_3d[:, depth_axis_index] < (z_min + edge_candidate_tolerance)
edge_candidate_points_3d = points_3d[candidate_mask]

# 可视化候选点（会看到一个倾斜的、有厚度的点带）
pcd_candidates = o3d.geometry.PointCloud()
pcd_candidates.points = o3d.utility.Vector3dVector(edge_candidate_points_3d)
# o3d.visualization.draw_geometries([pcd_final, pcd_candidates])

# ------ 2D投影与边缘检测 ------
from sklearn.linear_model import RANSACRegressor

# 1. 将3D候选点投影到XY平面
# X是自变量，Y是因变量。这里我们用X坐标预测Y坐标
points_2d = edge_candidate_points_3d[:, :2] # 只取X和Y
X = points_2d[:, 0].reshape(-1, 1) # X坐标
y = points_2d[:, 1]                # Y坐标

# 2. 初始化并运行 RANSAC 回归器
# residual_threshold: 样本点被认为是内点的最大距离阈值，需要根据点云密度调整
ransac = RANSACRegressor(residual_threshold=0.005) # 5mm的容忍度
ransac.fit(X, y)

# 3. 获取内点（即构成直线的主要点）
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 4. 提取出属于直线的3D点
line_points_3d = edge_candidate_points_3d[inlier_mask]

if line_points_3d.shape[0] < 2:
    print("未能找到足够的内点来确定线段！")
else:
    # 我们可以沿着X轴找到两个端点
    x_coords = line_points_3d[:, 0]
    
    # 找到X最小和最大的点的索引
    min_x_index = np.argmin(x_coords)
    max_x_index = np.argmax(x_coords)
    
    # 获取两个3D端点
    endpoint1 = line_points_3d[min_x_index]
    endpoint2 = line_points_3d[max_x_index]
    
    print(f"拟合出的线段端点1: {endpoint1}")
    print(f"拟合出的线段端点2: {endpoint2}")


grasp_point = (endpoint1 + endpoint2) / 2.0

print(f"最终计算出的抓取点坐标: {grasp_point}")

# 可视化最终结果
pcd_line_points = o3d.geometry.PointCloud()
pcd_line_points.points = o3d.utility.Vector3dVector(line_points_3d)
pcd_line_points.paint_uniform_color([1.0, 0, 0]) # 红色内点

grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
grasp_sphere.translate(grasp_point)
grasp_sphere.paint_uniform_color([0, 1.0, 0]) # 绿色抓取点

o3d.visualization.draw_geometries([pcd_pink_bag, pcd_line_points, grasp_sphere])
