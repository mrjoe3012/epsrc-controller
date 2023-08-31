from rclpy.node import Node
from ugrdv_msgs.msg import Cone3dArray, Cone3d, CarRequest
from eufs_msgs.msg import ConeArrayWithCovariance
from epsrc_controller import TrackNetwork, NodeType, Path
from epsrc_controller.utils import getMessageHash
from numpy.typing import NDArray
import numpy as np
import rclpy, math

class ControllerNode(Node):
    def __init__(self):
        """
        Parameters:
        - 'perception-cones-topic', string
        - 'car-request-topic', string
        - 'path-planning-beam-width', int
        """
        super().__init__("epsrc_controller")
        self.declare_parameter("perception-cones-topic", "")
        self.declare_parameter("car-request-topic", "")
        self.declare_parameter("path-planning-beam-width", 3)
        self.cones_topic = self.get_parameter("perception-cones-topic").value
        self.car_request_topic = self.get_parameter("car-request-topic").value
        self.beam_width = self.get_parameter("path-planning-beam-width").value
        self.cones_sub = self.create_subscription(Cone3dArray, self.cones_topic, self.on_ugr_cones, 1) 
        self.car_request_pub = self.create_publisher(CarRequest, self.car_request_topic, 1)

        self.wheel_base = 1.57
        self.max_steering_angle_magnitude = math.radians(21.0)
        self.info = lambda x: self.get_logger().info(x)

    def get_path_velocity(self, cost: float) -> float:
        """
        Get target velocity as a function of path cost.
        :param cost: Path cost
        :returns: velocity
        """
        velocity = 3.0
        if cost >= 3.5: # bad path ~2% of paths
            return velocity * 0.2
        elif cost >= 1.5:  # medium cost ~85%
            return velocity * 0.6
        else: # low cost path ~ 13% of paths
            return velocity

    def get_steering_angle_slowdown(self, steering_angle: float) -> float:
        """
        Get a multiplier as a function of the steering angle. Uses a parameterised
        exponential function to drive slower when steering harshly.
        :param: steering_angle: The steering angle in radians.
        :returns: A mutliplier in [0,1]
        """
        a = 3.2
        b = -0.134
        c = -2.2
        norm = min(abs(steering_angle) / self.max_steering_angle_magnitude, 1.0)
        multiplier = a * math.exp(b * norm) + c
        return multiplier

    def get_steering_angle(self, target_point: NDArray) -> float:
        """
        Uses the equations of the kinematic bicycle model to solve for a steering
        angle which intersects the front acle with a target point. Returns at most,
        the maximum steering angle achievable by the car.
        :param target_point: A (2,) coordinate target point.
        :returns: The steering angle.
        """
        ax_f = np.array([self.wheel_base, 0.0])
        tp = target_point
        # determine the steering direction
        if tp[1] == 0.0:
            return 0.0
        elif tp[1] > 0.0:
            direction = 1.0
        else:
            direction = -1.0
        icr = np.array([
            0.0,
            direction * (tp[0] + tp[1]) / (2.0 * tp[1])
        ])
        ax_f_vec = icr - ax_f
        tp_vec = icr - tp
        steering_angle_mag = math.acos(
            np.dot(ax_f_vec, tp_vec) \
            / (np.linalg.norm(ax_f_vec) * np.linalg.norm(tp_vec))
        )
        steering_angle_mag = min(steering_angle_mag, self.max_steering_angle_magnitude)
        steering_angle = direction * steering_angle_mag
        return steering_angle

    def on_ugr_cones(self, msg: Cone3dArray) -> None:
        """
        1. convert the cones into two numpy arrays, one containing the cone
        positions and one containing the cone colours
        2. construct a graph representation of the track and use it to retrieve
        a low cost path
        3. publish a CarRequest with the commands required to drive to the first
        point on the path
        """
        info = lambda x: self.get_logger().info(x)
        if len(msg.cones) < 2:
            info("Not enough input cones.")
            return
        colour2nodetype = {
            Cone3d.BLUE : NodeType.BLUE_CONE,
            Cone3d.YELLOW : NodeType.YELLOW_CONE,
            Cone3d.ORANGE : NodeType.ORANGE_CONE,
            Cone3d.LARGEORANGE : NodeType.LARGE_ORANGE_CONE,
            Cone3d.UNKNOWN : NodeType.UNKNOWN_CONE
        }
        pts = np.array([[cone.position.x, cone.position.y] for cone in msg.cones])
        node_types = [colour2nodetype[cone.colour] for cone in msg.cones]
        net = TrackNetwork(pts, node_types)
        beam_width = self.beam_width
        beam_iterations = max(len(pts) // 2, 1)
        paths = net.beam_search(beam_width, beam_iterations)
        if len(paths) == 0:
            info("No paths were returned.")
            return
        path_cost, best_path = paths[0]
        if len(best_path) < 1:
            info("Best path is empty.")
            return
        first_point = net.get_edge_vertex(best_path[0])
        car_request = CarRequest()
        car_request.header.stamp = self.get_clock().now().to_msg()
        car_request.velocity = self.get_path_velocity(path_cost)
        car_request.steering_angle = self.get_steering_angle(first_point)
        car_request.velocity *= self.get_steering_angle_slowdown(car_request.steering_angle)
        car_request.meta.hash = getMessageHash(car_request)
        car_request.meta.consumed_messages = [msg.meta.hash]
        self.car_request_pub.publish(car_request)

def main():
    rclpy.init()
    node = ControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()