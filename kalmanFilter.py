import numpy as np
from numpy import dot,sum,tile,linalg
from numpy.linalg import inv
import Kalman_Utilities

fjfjfjf
class kalman_filter:
    def __init__(self, initial_location, frames_per_second):
        global delta_t
        delta_t = 1 / frames_per_second
        # the mean state estimate of the previous step
        self.current_state = Kalman_Utilities.init_state_vector(initial_location)

        # the transition matrix
        self.state_transition = Kalman_Utilities.init_state_transition(frames_per_second)

        # the state covariance of previous step
        self.covariance = Kalman_Utilities.init_covariance_matrix()

        # the process noise covariance matrix
        self.process_noise_covariance = Kalman_Utilities.init_process_noise_covariance()

        # the measuring matrix
        self.measuring_matrix = Kalman_Utilities.init_measuring_matrix()

        # the measurement noise covariance matrix
        self.measurement_noise = Kalman_Utilities.init_measurement_noise()

    def projects(self):
        self.current_state = dot(self.state_transition, self.current_state)
        self.covariance = dot(self.state_transition, dot(self.covariance, self.state_transition.T)) + self.measurement_noise

    def update(self, measurement):
        IM = dot(self.measuring_matrix, self.current_state)
        IS = dot(self.measuring_matrix, dot(self.covariance, self.measuring_matrix.T)) + self.process_noise_convairance
        K = dot(self.covariance, dot(self.measuring_matrix.T, inv(IS)))
        self.current_state = self.current_state + dot(K, (measurement - IM))
        self.covariance = (np.identity(4) - K * self.measuring_matrix) * self.covariance

    def get_prediction(self, measurement):
        self.projects()
        self.update()
        return measurement - dot(self.measuring_matrix, self.current_state)
