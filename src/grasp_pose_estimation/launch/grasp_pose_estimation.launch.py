from launch import LaunchDescription
from launch.actions import OpaqueFunction, ExecuteProcess
from launch.substitutions import ThisLaunchFileDir
from launch.launch_context import LaunchContext
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os


def get_this_package_name(context: LaunchContext):
    this_launch_file_dir = ThisLaunchFileDir().perform(context)
    return os.path.basename(os.path.dirname(this_launch_file_dir))


def launch_setup(context, *args, **kwargs):
    launch_entities = []

    this_pkg_name = get_this_package_name(context)
    pkg_install_path = get_package_prefix(this_pkg_name)
    ws_path = os.path.dirname(os.path.dirname(pkg_install_path))
    python_path = os.path.join(ws_path, ".venv", "bin", "python")

    # print("This package name:", this_pkg_name)
    # print("Package install path:", pkg_install_path)
    # print("Workspace path:", ws_path)
    # print("Python path:", python_path)

    launch_entities.append(
        ExecuteProcess(
            name="groundedSAM_based_edge_estimation_node",
            cmd=[
                "HF_HUB_OFFLINE=1",
                python_path,
                os.path.join(
                    pkg_install_path, "lib", this_pkg_name, "groundedSAM_based_edge_estimation_node"
                ),
            ],
            output="screen",
            shell=True,
            emulate_tty=True,
        ),
    )

    return launch_entities


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(OpaqueFunction(function=launch_setup))
    return LaunchDescription(declared_arguments)
