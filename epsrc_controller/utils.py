from numpy.typing import NDArray
from rclpy.serialization import serialize_message
import numpy as np
import math, hashlib, copy

INT_T = np.int32
FLOAT_T = np.float32

def vector_angle(v1: NDArray, v2: NDArray) -> float:
    """
    Determines the smallest angle between two vectors, positive angles
    going counter-clockwise.
    :param v1: The first vector.
    :param v2: The second vector.
    :returns: Angle in radians.
    """
    return np.arctan2(
        np.linalg.det(np.vstack((v1, v2)).T),
        np.dot(v1, v2) 
    )

def vel_to_rpm(vel: float) -> float:
    wheel_radius = 0.26
    return (vel / (2.0 * math.pi * wheel_radius)) * 60.0

def getMessageHash(msg):
	"""
	Returns the MD5 hash for a message.
	
	:param msg: The message to calculate a hash for. Must have a ugrdv_msgs/Meta field named meta.
	:returns: A hex string containing the hash.
	"""
	msg = copy.deepcopy(msg)
	msg.meta.hash = ""
	serialized = serialize_message(msg)
	hash = hashlib.md5(serialized, usedforsecurity=False)
	return hash.hexdigest().upper()