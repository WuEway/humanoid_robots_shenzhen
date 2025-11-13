from setuptools import find_packages, setup
from glob import glob
import os

package_name = "grasp_pose_estimation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "srv"), glob("srv/*.srv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rog",
    maintainer_email="rog@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "groundedSAM_based_edge_estimation_node = grasp_pose_estimation.groundedSAM_based_edge_estimation_node:main",
            "groundedSAM_based_handle_estimation_node = grasp_pose_estimation.groundedSAM_based_handle_estimation_node:main",
            "ImageToPose_srv = grasp_pose_estimation.ImageToPose_srv:main",
            "image_to_grasp_client = grasp_pose_estimation.image_to_grasp_client:main",
            "yolo_based_edge_estimation_node = grasp_pose_estimation.YOLO_based_edge_estimation_node:main",
            "yolo_based_handle_estimation_node = grasp_pose_estimation.YOLO_based_handle_estimation_node:main",
            "yolo_edge_or_handle_srv = grasp_pose_estimation.YOLO_edge_or_handle_srv:main",
        ],
    },
)
