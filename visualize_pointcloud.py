#!/usr/bin/env python3
"""
点云可视化工具
用于可视化保存的点云文件，支持PLY和NPY格式
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path

# 尝试导入Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("❌ 未安装Open3D库，无法进行点云可视化")
    print("💡 安装方法: pip install open3d")


class PointCloudVisualizer:
    """点云可视化器"""
    
    def __init__(self):
        self.supported_formats = ['.ply', '.npy']
        
    def load_point_cloud_from_file(self, filepath: str) -> o3d.geometry.PointCloud:
        """从文件加载点云"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
            
        if filepath.suffix.lower() == '.ply':
            # 加载PLY文件
            pcd = o3d.io.read_point_cloud(str(filepath))
            if len(pcd.points) == 0:
                raise ValueError(f"PLY文件为空: {filepath}")
            return pcd
            
        elif filepath.suffix.lower() == '.npy':
            # 加载NPY文件
            data = np.load(filepath, allow_pickle=True)
            
            # 检查是否是新格式（包含颜色的字典）
            if isinstance(data, dict) and "points" in data:
                points = data["points"]
                colors = data.get("colors")
                
                if len(points) == 0:
                    raise ValueError(f"NPY文件为空: {filepath}")
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # 如果有颜色信息，添加颜色
                if colors is not None and len(colors) == len(points):
                    # 将颜色值从0-255范围转换到0-1范围
                    colors_normalized = colors.astype(np.float64) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
                    
                return pcd
            else:
                # 兼容旧格式（纯点云数组）
                points = data
                if len(points) == 0:
                    raise ValueError(f"NPY文件为空: {filepath}")
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                return pcd
            
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
    
    def enhance_point_cloud(self, pcd: o3d.geometry.PointCloud, 
                          colorize_by_depth: bool = True) -> o3d.geometry.PointCloud:
        """增强点云的视觉效果"""
        # 只有在没有颜色时才添加深度着色
        if not pcd.has_colors() and colorize_by_depth:
            points = np.asarray(pcd.points)
            if len(points) > 0:
                # 根据Z值着色
                z_values = points[:, 2]
                z_min, z_max = np.min(z_values), np.max(z_values)
                
                if z_max > z_min:
                    # 归一化Z值并映射到颜色
                    z_normalized = (z_values - z_min) / (z_max - z_min)
                    colors = np.zeros((len(points), 3))
                    
                    # 使用彩虹色映射：蓝(近) -> 绿 -> 红(远)
                    colors[:, 0] = np.clip(2 * z_normalized - 1, 0, 1)  # 红色
                    colors[:, 1] = np.clip(2 * (1 - np.abs(z_normalized - 0.5)), 0, 1)  # 绿色
                    colors[:, 2] = np.clip(2 * (1 - z_normalized), 0, 1)  # 蓝色
                    
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    # 统一颜色
                    pcd.paint_uniform_color([0.0, 0.7, 0.9])
        elif pcd.has_colors():
            print(f"🎨 使用原始颜色信息（{len(pcd.colors)}个点有颜色）")
        
        return pcd
    
    def visualize_point_cloud(self, filepath: str, enhance: bool = True):
        """可视化点云文件"""
        if not HAS_OPEN3D:
            print("❌ 无法可视化：未安装Open3D库")
            return
            
        try:
            print(f"🔄 加载点云文件: {os.path.basename(filepath)}")
            pcd = self.load_point_cloud_from_file(filepath)
            
            if enhance:
                pcd = self.enhance_point_cloud(pcd)
            
            points = np.asarray(pcd.points)
            print(f"📊 点云统计:")
            print(f"   - 点数: {len(points)}")
            print(f"   - 颜色: {'有' if pcd.has_colors() else '无'}")
            
            if len(points) > 0:
                center = np.mean(points, axis=0)
                print(f"   - 中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            
            print(f"🖼️  启动3D可视化窗口...")
            print(f"💡 操作提示:")
            print(f"   - 鼠标左键拖拽: 旋转视角")
            print(f"   - 鼠标右键拖拽: 平移视图") 
            print(f"   - 滚轮: 缩放")
            print(f"   - ESC键或关闭窗口: 退出")
            
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"点云可视化 - {os.path.basename(filepath)}",
                width=1024,
                height=768
            )
            
            # 添加点云
            vis.add_geometry(pcd)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = 2.0  # 点大小
            render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()

    def get_file_info(self, filepath: str) -> dict:
        """获取点云文件信息而不进行可视化"""
        try:
            pcd = self.load_point_cloud_from_file(filepath)
            
            points = np.asarray(pcd.points)
            info = {
                "filepath": filepath,
                "point_count": len(points),
                "has_colors": pcd.has_colors(),
                "file_size": os.path.getsize(filepath)
            }
            
            if len(points) > 0:
                # 计算边界框
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                center = np.mean(points, axis=0)
                
                info.update({
                    "min_coords": min_coords.tolist(),
                    "max_coords": max_coords.tolist(),
                    "center": center.tolist(),
                    "dimensions": (max_coords - min_coords).tolist()
                })
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_point_cloud_files(self, directory: str) -> list:
        """列出目录中的所有点云文件"""
        directory = Path(directory)
        if not directory.exists():
            return []
        
        files = []
        for ext in self.supported_formats:
            files.extend(directory.glob(f"*{ext}"))
        
        return sorted(files)
    
    def interactive_file_selection(self, directory: str = "detection_results"):
        """交互式文件选择和可视化"""
        files = self.list_point_cloud_files(directory)
        
        if not files:
            print(f"❌ 在目录 '{directory}' 中没有找到点云文件")
            print(f"💡 支持的格式: {', '.join(self.supported_formats)}")
            return None
        
        print(f"\n📂 在目录 '{directory}' 中找到 {len(files)} 个点云文件:")
        for i, file in enumerate(files):
            file_size = file.stat().st_size
            print(f"  {i+1:2d}. {file.name} ({file_size} bytes)")
        
        while True:
            try:
                choice = input(f"\n请选择文件编号 (1-{len(files)}), 'a' 显示所有信息, 或 'q' 退出: ").strip().lower()
                
                if choice == 'q' or choice == 'quit':
                    print("👋 退出选择")
                    return None
                
                if choice == 'a' or choice == 'all':
                    # 显示所有文件信息
                    for file in files:
                        print(f"\n📄 {file.name}:")
                        info = self.get_file_info(str(file))
                        if "error" in info:
                            print(f"   ❌ 错误: {info['error']}")
                        else:
                            print(f"   📊 点数: {info['point_count']}")
                            print(f"   🎨 颜色: {'有' if info['has_colors'] else '无'}")
                            if "center" in info:
                                center = info["center"]
                                print(f"   📍 中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                    
                    # 重新显示文件列表供用户选择
                    print(f"\n📂 在目录 '{directory}' 中找到 {len(files)} 个点云文件:")
                    for i, file in enumerate(files):
                        file_size = file.stat().st_size
                        print(f"  {i+1:2d}. {file.name} ({file_size} bytes)")
                    continue
                
                index = int(choice) - 1
                if 0 <= index < len(files):
                    selected_file = files[index]
                    
                    # 询问用户操作
                    action = input(f"\n选择操作 - [v]可视化, [i]信息, [b]返回: ").strip().lower()
                    
                    if action == 'v' or action == 'visualize':
                        self.visualize_point_cloud(str(selected_file))
                    elif action == 'i' or action == 'info':
                        print(f"\n📄 {selected_file.name}:")
                        info = self.get_file_info(str(selected_file))
                        if "error" in info:
                            print(f"   ❌ 错误: {info['error']}")
                        else:
                            print(f"   📊 点数: {info['point_count']}")
                            print(f"   🎨 颜色: {'有' if info['has_colors'] else '无'}")
                            print(f"   💾 大小: {info['file_size']} bytes")
                            if "center" in info:
                                center = info["center"]
                                dimensions = info["dimensions"]
                                print(f"   📍 中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                                print(f"   📏 尺寸: {dimensions[0]:.3f} × {dimensions[1]:.3f} × {dimensions[2]:.3f} m")
                    elif action == 'b' or action == 'back':
                        continue
                    else:
                        print("❌ 无效操作，请输入 v, i 或 b")
                        
                else:
                    print(f"❌ 请输入 1-{len(files)} 之间的数字")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n👋 退出选择")
                return None


def main():
    parser = argparse.ArgumentParser(
        description="点云可视化和信息查看工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化单个文件
  python3 visualize_pointcloud.py file.ply
  
  # 查看文件信息（不可视化）
  python3 visualize_pointcloud.py --info file.ply
  
  # 交互式选择文件
  python3 visualize_pointcloud.py -i
  
  # 从指定目录交互选择
  python3 visualize_pointcloud.py -i -d /path/to/pointclouds
        """
    )
    
    parser.add_argument('files', nargs='*', help='要处理的点云文件路径')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='交互式文件选择模式')
    parser.add_argument('-d', '--directory', default='detection_results',
                       help='搜索点云文件的目录 (默认: detection_results)')
    parser.add_argument('--info', action='store_true',
                       help='只显示文件信息，不进行可视化')
    
    args = parser.parse_args()
    
    if not HAS_OPEN3D:
        print("⚠️ 未安装Open3D，只能提供基本文件信息")
    
    visualizer = PointCloudVisualizer()
    
    # 交互模式
    if args.interactive:
        visualizer.interactive_file_selection(args.directory)
        return 0
    
    # 命令行指定文件
    elif args.files:
        for filepath in args.files:
            if not os.path.exists(filepath):
                print(f"❌ 文件不存在: {filepath}")
                continue
                
            if args.info:
                # 只显示信息
                print(f"\n📄 文件: {os.path.basename(filepath)}")
                if HAS_OPEN3D:
                    info = visualizer.get_file_info(filepath)
                    if "error" in info:
                        print(f"   ❌ 错误: {info['error']}")
                    else:
                        print(f"   📊 点数: {info['point_count']}")
                        print(f"   🎨 颜色: {'有' if info['has_colors'] else '无'}")
                        print(f"   💾 大小: {info['file_size']} bytes")
                        if "center" in info:
                            center = info["center"]
                            dimensions = info["dimensions"]
                            print(f"   📍 中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                            print(f"   📏 尺寸: {dimensions[0]:.3f} × {dimensions[1]:.3f} × {dimensions[2]:.3f} m")
                else:
                    file_size = os.path.getsize(filepath)
                    print(f"   💾 大小: {file_size} bytes")
            else:
                # 直接可视化
                visualizer.visualize_point_cloud(filepath)
    
    # 默认：显示目录中的文件列表
    else:
        files = visualizer.list_point_cloud_files(args.directory)
        if not files:
            print(f"❌ 在目录 '{args.directory}' 中没有找到点云文件")
            print("💡 使用 -h 查看帮助信息")
            return 1
        
        print(f"📂 目录 '{args.directory}' 中的点云文件:")
        for file in files:
            file_size = file.stat().st_size
            print(f"   📄 {file.name} ({file_size/1024:.1f} KB)")
        
        print(f"\n💡 使用方法:")
        print(f"   - 可视化文件: python3 visualize_pointcloud.py <文件名>")
        print(f"   - 交互模式: python3 visualize_pointcloud.py -i")
        print(f"   - 查看信息: python3 visualize_pointcloud.py --info <文件名>")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
