from ugrdv_msgs.msg import VCUStatus
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.logging import get_logger
from epsrc_controller import utils
import numpy as np
import math

class PIDParams:
    def __init__(self, k_p: float, k_i: float,
                 k_d: float, window_size: float,
                 derivative_time_slice: float,
                 max_i_term: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window_size = window_size
        self.derivative_time_slice = derivative_time_slice
        self.max_i_term = max_i_term

class PID:
    def __init__(self, params):
        self._latest_time = 0.0
        self._states = []
        self._params = params
        self._target_rpm = 0.0
        self._gains = np.array([
            self._params.k_p,
            self._params.k_i,
            self._params.k_d
        ])
        self._wheel_radius = 0.26
        self._logger = get_logger("pid")
        self.info = lambda x: self._logger.info(x)

    def _make_state(self, time: float, rpm_error: float) -> dict:
        return {
            'time' : time,
            'rpm_error' : rpm_error
        }

    def _prune_states(self, time: float, states: list[dict]) -> list[dict]:
        i = 0
        window_size = self._params.window_size
        while time - states[i]["time"] > window_size:
            i += 1
        if i == 0:
            return states
        else:
            return states[i:]

    def _calculate_derivative(self, states: list[dict]) -> float:
        d_t = self._params.derivative_time_slice
        sum_err = 0.0
        time = self._latest_time
        i = len(states) - 1
        j = 0
        while i >= 0:
            state = states[i]
            if time - state["time"] > d_t: break
            sum_err += state["rpm_error"]
            i -= 1
        return sum_err / d_t

    def _calculate_integral(self, states: list[dict]) -> float:
        integral = 0.0
        for i in range(len(states) - 1):
            s1 = states[i]
            s2 = states[i+1]
            dt = s2["time"] - s2["time"]
            derr = s2["rpm_error"] - s2["rpm_error"]
            integral += dt * derr
        return integral

    def set_target_velocity(self, target: float) -> None:
        self._target_rpm = max(0.0, utils.vel_to_rpm(target))

    def update(self, time: float, vcu_status: VCUStatus) -> float:
        if time < self._latest_time: return
        self._latest_time = time
        rpm = np.mean([
            vcu_status.wheel_speeds.fl_speed,
            vcu_status.wheel_speeds.fr_speed,
            vcu_status.wheel_speeds.rl_speed,
            vcu_status.wheel_speeds.rr_speed
        ])
        rpm_error = self._target_rpm - rpm
        self._states.append(self._make_state(
            time=time,
            rpm_error=rpm_error
        ))
        self._states = self._prune_states(time, self._states)
        rpm_deriv = self._calculate_derivative(self._states)
        rpm_integral = self._calculate_integral(self._states)
        terms = np.array([
            rpm_error,
            rpm_integral,
            rpm_deriv
        ])
        terms *= self._gains
        terms[1] = max(-self._params.max_i_term, min(self._params.max_i_term, terms[1]))
        # self.info(f"{terms=}")
        return np.sum(terms)