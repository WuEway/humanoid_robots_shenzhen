from launch import LaunchDescription
from launch.actions import OpaqueFunction, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import ThisLaunchFileDir
from launch.launch_context import LaunchContext
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
from launch_ros.actions import Node


def get_this_package_name(context: LaunchContext):
    this_launch_file_dir = ThisLaunchFileDir().perform(context)
    return os.path.basename(os.path.dirname(this_launch_file_dir))


def launch_setup(context, *args, **kwargs):
    launch_entities = []

    this_pkg_name = get_this_package_name(context)
    pkg_install_path = get_package_prefix(this_pkg_name)
    ws_path = os.path.dirname(os.path.dirname(pkg_install_path))
    python_path = os.path.join(ws_path, ".venv", "bin", "python")

    launch_entities.append(
        SetEnvironmentVariable(
            name="HF_HUB_OFFLINE",
            value="1",
        )
    )

    launch_entities.append(
        Node(
            prefix=f"{python_path}",
            package=this_pkg_name,
            executable="yolo_based_handle_estimation_node",
            output="screen",
            emulate_tty=True,
        )
    )

    return launch_entities


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(OpaqueFunction(function=launch_setup))
    return LaunchDescription(declared_arguments)
