from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    param_info = [
        ("perception_model", "realistic"),
        ("drive_request_topic", "/drive_request"),
        ("car_request_topic", "/car_request"),
        ("ackermann_topic", "/cmd"),
        ("gt_cones_topic", "/ground_truth/track"),
        ("perception_cones_topic", "/perception_cones"),
        ("perception_cones_vis_topic", "/perception_cones/vis"),
        ("simulated_cones_topic", "/simulated_cones"),
        ("wheel_speeds_topic", "/ros_can/wheel_speeds"),
        ("vcu_status_topic", "/vcu_status"),
        ("gt_car_state_topic", "/ground_truth/state"),
        ("path_planning_beam_width", "3"),
        ("use_sim_time", "true"),
        ("sensor_fov", "180.0"),
        ("sensor_range", "12.0"),
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
            (str, 'perception_cones_topic'),
            (str, 'car_request_topic'),
            (int, 'path_planning_beam_width'),
        ]

        communicator_params = [
            (bool, 'use_sim_time'),
            (str, 'drive_request_topic'),
            (str, 'car_request_topic'),
            (str, 'ackermann_topic'),
            (str, 'perception_cones_topic'),
            (str, 'simulated_cones_topic'),
            (str, 'perception_cones_vis_topic'),
            (str, 'wheel_speeds_topic'),
            (str, 'vcu_status_topic'),
        ]

        simulated_perception_params = [
            (bool, 'use_sim_time'),
            (str, 'gt_cones_topic'),
            (str, 'gt_car_state_topic'),
            (str, 'simulated_cones_topic'),
            (str, 'perception_model'),
            (float, 'sensor_range'),
            (float, 'sensor_fov'),
        ]

        def get_params(d):
            return {
                p : f(param_values[p]) for f, p in d
            }

        return [LaunchDescription([
            Node(
                package='epsrc_controller',
                executable='controller',
                parameters=[
                    get_params(controller_params)
                ]
            ),
            Node(
                package='epsrc_controller',
                executable='communicator',
                parameters=[
                    get_params(communicator_params)
                ]
            ),
            Node(
                package='sim_data_collection',
                executable='perception_model',
                parameters=[
                    get_params(simulated_perception_params)
                ]
            )
        ])]

    return LaunchDescription(
        launch_args + [
            OpaqueFunction(function=nodes)
        ]
    )