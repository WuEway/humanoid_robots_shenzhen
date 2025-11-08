"""Record images from a ROS2 camera topic into a video file."""

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoRecorder(Node):
	def __init__(
		self,
		topic: str,
		output_path: Path,
		fps: float,
		frame_limit: int | None,
		duration: float | None,
		codec: str,
	) -> None:
		super().__init__("rgb_video_recorder")
		self.topic = topic
		self.output_path = output_path
		self.fps = fps
		self.frame_limit = frame_limit
		self.duration = duration
		self.codec = codec

		self.bridge = CvBridge()
		self.subscription = self.create_subscription(Image, topic, self._image_callback, 10)

		self.video_writer: cv2.VideoWriter | None = None
		self.frame_count = 0
		self.start_time = None
		self._stop_requested = False

	def _image_callback(self, msg: Image) -> None:
		if self.start_time is None:
			self.start_time = self.get_clock().now()

		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except Exception as exc:  # noqa: BLE001
			self.get_logger().error(f"Failed to convert image: {exc}")
			return

		if self.video_writer is None:
			height, width = frame.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*self.codec)
			self.video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
			if not self.video_writer.isOpened():
				self.get_logger().error("Could not open video writer; stopping.")
				self._request_stop()
				return
			self.get_logger().info(
				f"Recording {width}x{height} @ {self.fps:.2f} FPS from '{self.topic}' -> {self.output_path}"
			)

		self.video_writer.write(frame)
		self.frame_count += 1

		if self.frame_limit is not None and self.frame_count >= self.frame_limit:
			self.get_logger().info("Frame limit reached; stopping recording.")
			self._request_stop()

		if self.duration is not None and self.start_time is not None:
			elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
			if elapsed >= self.duration:
				self.get_logger().info("Duration reached; stopping recording.")
				self._request_stop()

	def _request_stop(self) -> None:
		self._stop_requested = True

	def stop_requested(self) -> bool:
		return self._stop_requested

	def cleanup(self) -> None:
		if self.video_writer is not None:
			self.video_writer.release()
		self.get_logger().info(f"Captured {self.frame_count} frames.")


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--topic",
		default="/woosh/camera/woosh_left_hand_rgbd/color/image_raw",
		help="Camera topic to record.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output video path (.mp4 recommended). Defaults to videos/<timestamp>.mp4",
	)
	parser.add_argument("--fps", type=float, default=30.0, help="Target video frame rate.")
	parser.add_argument(
		"--duration",
		type=float,
		default=None,
		help="Stop after this many seconds (omit to run until interrupted).",
	)
	parser.add_argument(
		"--frames",
		type=int,
		default=None,
		help="Stop after this many frames (checked in addition to duration).",
	)
	parser.add_argument(
		"--codec",
		default="mp4v",
		help="FourCC codec passed to OpenCV (default: mp4v).",
	)
	return parser.parse_known_args(argv)


def resolve_output_path(path_arg: Path | None) -> Path:
	if path_arg is not None:
		target = path_arg
	else:
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		target = Path("videos") / f"recording_{timestamp}.mp4"
	target.parent.mkdir(parents=True, exist_ok=True)
	return target


def main(argv: list[str] | None = None) -> None:
	parsed_args, ros_args = parse_args(argv)
	output_path = resolve_output_path(parsed_args.output)

	rclpy.init(args=ros_args)
	node = VideoRecorder(
		topic=parsed_args.topic,
		output_path=output_path,
		fps=parsed_args.fps,
		frame_limit=parsed_args.frames,
		duration=parsed_args.duration,
		codec=parsed_args.codec,
	)

	try:
		while rclpy.ok() and not node.stop_requested():
			rclpy.spin_once(node, timeout_sec=0.1)
	except KeyboardInterrupt:
		node.get_logger().info("Interrupted by user; stopping recording.")
	finally:
		node.cleanup()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
