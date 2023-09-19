from rclpy.node import Node
from ugrdv_msgs.msg import DriveRequest, Cone3dArray, Cone3d, VCUStatus, CarRequest
from eufs_msgs.msg import ConeArrayWithCovariance, CarState, WheelSpeedsStamped
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import MarkerArray, Marker
from copy import deepcopy
from epsrc_controller import PID, PIDParams
from epsrc_controller.utils import getMessageHash
from ament_index_python import get_package_share_directory
import numpy as np
import rclpy, math

class CommunicatorNode(Node):
    def __init__(self):
        super().__init__("epsrc_communicator")
        self.declare_parameter("drive_request_topic", "")
        self.declare_parameter("car_request_topic", "")
        self.declare_parameter("ackermann_topic", "")
        self.declare_parameter("simulated_cones_topic", "")
        self.declare_parameter("perception_cones_topic", "")
        self.declare_parameter("wheel_speeds_topic", "")
        self.declare_parameter("vcu_status_topic", "")
        self.declare_parameter("perception_cones_vis_topic", "")
        self.drive_request_topic = self.get_parameter("drive_request_topic").value
        self.car_request_topic = self.get_parameter("car_request_topic").value
        self.ackermann_topic = self.get_parameter("ackermann_topic").value
        self.simulated_cones_topic = self.get_parameter("simulated_cones_topic").value
        self.perception_cones_topic = self.get_parameter("perception_cones_topic").value
        self.wheel_speeds_topic = self.get_parameter("wheel_speeds_topic").value
        self.vcu_status_topic = self.get_parameter("vcu_status_topic").value
        self.perception_cones_vis_topic = self.get_parameter("perception_cones_vis_topic").value
        self.drive_pub = self.create_publisher(DriveRequest, self.drive_request_topic, 1)
        self.car_request_sub = self.create_subscription(CarRequest, self.car_request_topic, self.on_car_request, 1)
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_topic, 1)
        self.simulated_cones_sub = self.create_subscription(ConeArrayWithCovariance, self.simulated_cones_topic, self.on_simulated_cones, 1)
        self.perception_cones_pub = self.create_publisher(Cone3dArray, self.perception_cones_topic, 1)
        self.wheel_speeds_sub = self.create_subscription(WheelSpeedsStamped, self.wheel_speeds_topic, self.on_wheel_speeds, 1)
        self.vcu_status_pub = self.create_publisher(VCUStatus, self.vcu_status_topic, 1)
        self.perception_cones_vis_pub = self.create_publisher(MarkerArray, self.perception_cones_vis_topic, 1)

        self.last_marker_count = 0
        self.small_cone_mesh = "file://" + get_package_share_directory("eufs_tracks") + "/meshes/cone.dae"
        self.large_cone_mesh = "file://" + get_package_share_directory("eufs_tracks") + "/meshes/cone_big.dae"
        pid_params = PIDParams(
            0.075,
            0.1,
            0.0,
            5.0,
            0.5,
            0.5
        )
        self.pid = PID(pid_params)

        self.latest_vcu_status = None
        self.last_car_state = None
        self.marker_pub = self.create_publisher(MarkerArray, "/markers", 1)
        self.info = lambda x: self.get_logger().info(x)

    def on_car_state(self, msg: CarState) -> None:
        """
        Car states contain the ground truth car pose from the simulation.
        """
        self.last_car_state = msg

    def on_car_request(self, msg: CarRequest) -> None:
        """
        1. use PID to get acceleration command
        2. publish ackermann drive message which can be used for control
        by the simulation
        3. publish drive request which contains meta information
        """
        if self.latest_vcu_status is None: return
        self.pid.set_target_velocity(msg.velocity)
        time = self.get_clock().now().nanoseconds / 1e9
        acceleration = self.pid.update(time, self.latest_vcu_status)
        ackermann = AckermannDriveStamped()
        ackermann.header.stamp = self.get_clock().now().to_msg()
        ackermann.drive.steering_angle = msg.steering_angle
        ackermann.drive.acceleration = acceleration
        self.ackermann_pub.publish(ackermann)
        drive = DriveRequest()
        drive.ackermann = ackermann
        drive.meta.hash = getMessageHash(drive)
        drive.meta.consumed_messages = [self.latest_vcu_status.meta.hash, msg.meta.hash]
        self.drive_pub.publish(drive)

    def gt_cones_to_local_frame(self, car_state: CarState, cones: ConeArrayWithCovariance) -> None:
        """
        Transform an array of ground truth cones so that they are in the car's frame of reference. This frame of reference
        is described by the pose within car_state.
        :param car_state: The car pose to use for transformation.
        :param cones: The cones to transform. Modified in-place.
        """
        car_pos = np.array([
            car_state.pose.pose.position.x,
            car_state.pose.pose.position.y
        ])
        orientation = Rotation.from_quat([
            car_state.pose.pose.orientation.x,
            car_state.pose.pose.orientation.y,
            car_state.pose.pose.orientation.z,
            car_state.pose.pose.orientation.w,
        ])
        yaw = -orientation.as_euler("XYZ")[2]
        rotation_matrix = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)]
        ])
        all_cones = \
            cones.blue_cones + cones.yellow_cones \
            + cones.orange_cones + cones.big_orange_cones \
            + cones.unknown_color_cones
        for cone in all_cones:
            new_pos = rotation_matrix @ (np.array([cone.point.x, cone.point.y]) - car_pos)
            cone.point.x, cone.point.y = new_pos[0], new_pos[1]

    def crop_to_fov(self, fov: float, range: float, cones: Cone3dArray) -> Cone3dArray:
        """
        Crop a cone array such that all cones lie within an angular FOV and are within a maximum range from the origin.
        :param fov: The angle in radians to use for the field of view.
        :param range: The maximum distance in metres.
        :param cones: The cones to crop.
        :returns: A new cone array containing only the cones passsing the crop constraints.
        """
        sqr_range = math.pow(range, 2)
        half_fov = fov / 2.0
        new_msg = deepcopy(cones)
        new_msg.cones = []
        for cone in cones.cones:
            pos = np.array([
                cone.position.x, cone.position.y
            ])
            angle = abs(np.arctan2(pos[1], pos[0]))
            sqr_dist = np.dot(pos, pos)
            if angle <= half_fov and sqr_dist <= sqr_range:
                new_msg.cones.append(cone)
        return new_msg

    def on_simulated_cones(self, msg: ConeArrayWithCovariance) -> None:
        """
        1. convert message from eufs format into ugrdv format and publish
        3. publish visualisation of perception cones
        """
        positions_and_colours = \
        [
            (Cone3d.BLUE, cone.point) for cone in msg.blue_cones
        ] + \
        [
            (Cone3d.YELLOW, cone.point) for cone in msg.yellow_cones
        ] + \
        [
            (Cone3d.ORANGE, cone.point) for cone in msg.orange_cones
        ] + \
        [
            (Cone3d.LARGEORANGE, cone.point) for cone in msg.big_orange_cones
        ] + \
        [
            (Cone3d.UNKNOWN, cone.point) for cone in msg.unknown_color_cones
        ]
        cones = Cone3dArray()
        cones.header.stamp = self.get_clock().now().to_msg()
        cones.header.frame_id = "base_footprint"
        for colour, point in positions_and_colours:
            cone = Cone3d()
            cone.position.x = point.x
            cone.position.y = point.y
            cone.position.z = 0.0
            cone.colour = colour
            cones.cones.append(cone)
        cones.meta.hash = getMessageHash(cones)
        self.perception_cones_pub.publish(cones)
        self.visualise_cones(cones)

    def visualise_cones(self, cones: Cone3dArray) -> None:
        marker_array = MarkerArray()
        i = 0
        colours = {
            Cone3d.BLUE : (0.0, 0.0, 1.0, 1.0),
            Cone3d.YELLOW : (1.0, 1.0, 0.0, 1.0),
            Cone3d.ORANGE : (1.0, 0.5, 0.0, 1.0),
            Cone3d.LARGEORANGE : (1.0, 0.5, 0.0, 1.0),
            Cone3d.UNKNOWN : (0.5, 0.5, 0.5, 1.0)
        }
        for cone in cones.cones:
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "base_footprint"
            marker.action = Marker.ADD
            marker.id = i
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = colours[cone.colour]
            marker.scale.x, marker.scale.y, marker.scale.z = 1.0, 1.0, 1.0
            marker.pose.position.x = cone.position.x
            marker.pose.position.y = cone.position.y
            marker.pose.position.z = 0.0
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = self.large_cone_mesh if cone.colour == Cone3d.LARGEORANGE else self.small_cone_mesh
            marker_array.markers.append(marker)
            i += 1
        for j in range(i, self.last_marker_count):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "base_footprint"
            marker.id = j
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.last_marker_count = i
        self.perception_cones_vis_pub.publish(marker_array)

    def get_front_wheel_speed(self, rear_wheel_speed: float, steering_angle: float) -> float:
        """
        Uses the kinematic bicycle model equations to get the speed of the front wheels given
        some rear wheel speed and steering angle.
        :param rear_wheel_speed: The speed of the rear wheel, in RPM or m/s.
        :param steering_angle: The car's steering angle in radians.
        :returns: The front wheel speed in the same unit as the rear wheel speed.
        """
        alpha = (np.pi / 2.0) - abs(steering_angle)
        front_wheel_speed = rear_wheel_speed * (math.sqrt(1.0 + math.pow(math.tan(alpha), 2)) / (math.tan(alpha)))
        print(f"{front_wheel_speed=}")
        return front_wheel_speed

    def on_wheel_speeds(self, msg: WheelSpeedsStamped) -> None:
        """
        1. obtain correct front wheel speeds
        2. publish a VCUStatus message containing steering angle and wheel speeds
        """
        steering_angle = msg.speeds.steering
        rl_speed = msg.speeds.lb_speed
        rr_speed = msg.speeds.rb_speed
        fl_speed = self.get_front_wheel_speed(rl_speed, steering_angle)
        fr_speed = self.get_front_wheel_speed(rr_speed, steering_angle)
        vcu = VCUStatus()
        vcu.header.stamp = self.get_clock().now().to_msg()
        vcu.steering_angle = steering_angle
        vcu.wheel_speeds.fl_speed = fl_speed
        vcu.wheel_speeds.fr_speed = fr_speed
        vcu.wheel_speeds.rl_speed = rl_speed
        vcu.wheel_speeds.rr_speed = rr_speed
        vcu.meta.hash = getMessageHash(vcu)
        self.vcu_status_pub.publish(vcu)
        self.latest_vcu_status = vcu
         

def main():
    rclpy.init()
    node = CommunicatorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()