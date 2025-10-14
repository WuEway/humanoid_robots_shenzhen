import numpy as np
import open3d as o3d # 导入 Open3D 库

# --- 你的原始代码 ---
# 加载保存的数据文件
# 请确保 'debug_pointcloud_1760439901.npz' 文件存在于当前目录或提供完整路径
file_name = 'debug_pointcloud_1760439901.npz'
data = np.load(file_name)

# 提取点和颜色
points_robot = data['points']
colors_cam = data['colors']

print(f"成功加载点云 '{file_name}'，包含 {len(points_robot)} 个点。")
# --- 原始代码结束 ---
from .grasp_pose_estimator import GraspPoseEstimator

grasp_estimator = GraspPoseEstimator(visualize=True)

grasp_pose_result = grasp_estimator.calculate_grasp_pose(points_robot, colors_cam)



# --- Open3D 可视化部分 ---

# 1. 创建一个 Open3D 的点云对象
pcd = o3d.geometry.PointCloud()

# 2. 将 NumPy 数组的点数据赋值给点云对象
# Open3D 期望点数据是 (N, 3) 的浮点数数组
pcd.points = o3d.utility.Vector3dVector(points_robot)

# 3. 将 NumPy 数组的颜色数据赋值给点云对象
# Open3D 期望颜色数据是 (N, 3) 的浮点数数组，且颜色值范围在 [0, 1] 之间。
# 如果你的 colors_cam 是 [0, 255] 范围的整数，需要进行归一化。
if colors_cam.dtype == np.uint8:
    colors_cam_normalized = colors_cam.astype(np.float64) / 255.0
else:
    colors_cam_normalized = colors_cam # 如果已经是 [0, 1] 的浮点数则直接使用

pcd.colors = o3d.utility.Vector3dVector(colors_cam_normalized)

# 4. 可视化点云
print("正在打开 Open3D 可视化窗口...")
# o3d.visualization.draw_geometries([pcd],
#                                   window_name="Robot Point Cloud Visualization",
#                                   width=800, height=600,
#                                   left=50, top=50,
#                                   mesh_show_back_face=False) # 针对mesh的选项，点云可以忽略

print("可视化完成。")
