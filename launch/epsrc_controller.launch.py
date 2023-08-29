from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    param_info = [
        ("drive-request-topic", "/drive_request"),
        ("car-request-topic", "/car_request"),
        ("ackermann-topic", "/cmd"),
        ("gt-cones-topic", "/ground_truth/track"),
        ("perception-cones-topic", "/perception_cones"),
        ("wheel-speeds-topic", "/ros_can/wheel_speeds"),
        ("vcu-status-topic", "/vcu_status"),
        ("gt-car-state-topic", "/ground_truth/state"),
        ("path-planning-beam-width", "3"),
        ("use_sim_time", "true"),
        ("sensor-fov", "180.0"),
        ("sensor-range", "12.0"),
    ]
    launch_args = [
        DeclareLaunchArgument(param_name, default_value=default) for \
            param_name, default in param_info
    ]

    def nodes(context, *args, **kwargs):
        param_values = {
            param_name : LaunchConfiguration(param_name).perform(context) \
                for param_name, _ in param_info
        }

        controller_params = [
            (bool, 'use_sim_time'),
            (str, 'perception-cones-topic'),
            (str, 'car-request-topic'),
            (int, 'path-planning-beam-width'),
        ]

        communicator_params = [
            (bool, 'use_sim_time'),
            (str, 'drive-request-topic'),
            (str, 'car-request-topic'),
            (str, 'ackermann-topic'),
            (str, 'gt-cones-topic'),
            (str, 'perception-cones-topic'),
            (str, 'wheel-speeds-topic'),
            (str, 'vcu-status-topic'),
            (str, 'gt-car-state-topic'),
            (float, 'sensor-range'),
            (float, 'sensor-fov'),
        ]

        return [LaunchDescription([
            Node(
                package='epsrc_controller',
                executable='controller',
                parameters=[{
                    p : f(param_values[p]) for f, p in controller_params
                }]
            ),
            Node(
                package='epsrc_controller',
                executable='communicator',
                parameters=[{
                    p : f(param_values[p]) for f, p in communicator_params
                }]
            ),
        ])]

    return LaunchDescription(
        launch_args + [
            OpaqueFunction(function=nodes)
        ]
    )