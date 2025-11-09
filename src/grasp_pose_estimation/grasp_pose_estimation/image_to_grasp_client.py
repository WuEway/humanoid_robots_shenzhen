import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Lock  
from image_to_grasp.srv import ImageToGrasp


class ImageSubAndGraspClient(Node):
    def __init__(self):
        super().__init__("image_sub_grasp_client")
        
        # 1. 初始化图像订阅相关
        self.bridge = CvBridge()
        self.mutex = Lock() 
        self.latest_color = None  # 缓存最新RGB图像
        self.latest_depth = None  # 缓存最新深度图像
        self.image_ready = False  # 标记图像是否已就绪
        
        # 2. 订阅RGB和深度图像话题（替换为你的相机话题）
        self.color_sub = self.create_subscription(
            Image,
            "/nbman/camera/nbman_head_rgbd/color/image_raw",
            self.color_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/nbman/camera/nbman_head_rgbd/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10
        )
        self.get_logger().info("已订阅相机图像话题，等待图像数据...")
        
        # 3. 初始化服务客户端（连接服务端）
        self.grasp_client = self.create_client(
            ImageToGrasp,
            "/grounding_dino/image_to_grasp"  # 服务话题（需与服务端一致）
        )
        # 等待服务端启动
        while not self.grasp_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("服务端未就绪，等待中...")
        
        # 4. 定时发送请求（每3秒一次）
        self.timer = self.create_timer(3.0, self.send_grasp_request)

    def color_callback(self, msg: Image):
        """RGB图像回调: 缓存最新帧"""
        with self.mutex:  # 加锁保证线程安全
            try:
                # 转换ROS图像→OpenCV格式（BGR8）
                self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.check_image_ready()  # 检查图像是否已齐全
            except Exception as e:
                self.get_logger().error(f"RGB图像转换失败: {str(e)}")

    def depth_callback(self, msg: Image):
        """深度图像回调：缓存最新帧"""
        with self.mutex:  # 加锁保证线程安全
            try:
                # 深度图像通常为16位单通道（单位：mm）
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.check_image_ready()  # 检查图像是否已齐全
            except Exception as e:
                self.get_logger().error(f"深度图像转换失败: {str(e)}")

    def check_image_ready(self):
        """检查RGB和深度图像是否都已缓存, 标记为就绪"""
        if self.latest_color is not None and self.latest_depth is not None:
            self.image_ready = True
        else:
            self.image_ready = False

    def send_grasp_request(self):
        """发送图像请求到服务端，获取抓取位姿"""
        with self.mutex:  # 加锁读取图像，避免数据冲突
            # 1. 检查图像是否就绪
            if not self.image_ready:
                self.get_logger().warn("图像未就绪(缺少RGB或深度),跳过请求")
                return
            
            # 2. 转换OpenCV图像→ROS消息（用于服务请求）
            color_msg = self.bridge.cv2_to_imgmsg(self.latest_color, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(self.latest_depth, "16UC1")
        
        # 3. 构建服务请求
        request = ImageToGrasp.Request()
        request.color_image = color_msg
        request.depth_image = depth_msg
        
        # 4. 异步发送请求（避免阻塞客户端）
        self.future = self.grasp_client.call_async(request)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        """处理服务端的响应结果"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"\n✅ 服务调用成功: {response.message}")
                self.print_grasp_pose(response.grasp_pose)
            else:
                self.get_logger().error(f"❌ 服务调用失败: {response.message}")
        except Exception as e:
            self.get_logger().error(f"服务响应处理失败: {str(e)}")

    def print_grasp_pose(self, pose: Pose):
        """打印抓取位姿详情"""
        self.get_logger().info("📌 抓取位姿详情:")
        self.get_logger().info(f"位置 (x,y,z): ({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f})")
        self.get_logger().info(f"姿态 (x,y,z,w): ({pose.orientation.x:.4f}, {pose.orientation.y:.4f}, "
                              f"{pose.orientation.z:.4f}, {pose.orientation.w:.4f})")
        self.get_logger().info("查看TF变换命令: ros2 run tf2_ros tf2_echo woosh_base_link grasp_food_pos")


def main(args=None):
    rclpy.init(args=args)
    client = ImageSubAndGraspClient()
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info("👋 用户中断，客户端退出")
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    