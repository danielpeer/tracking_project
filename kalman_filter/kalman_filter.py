from numpy import dot
from numpy.linalg import inv
from kalman_filter.kalman_utilities import *

class kalman_filter:
    def __init__(self, initial_location, frames_per_second):

        init_global_variables(frames_per_second)

        # the mean state estimate of the previous step
        self.current_state = init_state_vector(initial_location)

        # the transition matrix
        self.state_transition = init_state_transition(frames_per_second)

        # the state covariance of previous step
        self.covariance = init_covariance_matrix()

        # the process noise covariance matrix
        self.process_noise_covariance = init_process_noise_covariance()

        # the measuring matrix
        self.measuring_matrix = init_measuring_matrix()

        # the measurement noise covariance matrix
        self.measurement_noise = init_measurement_noise()

    def projects(self):
        self.current_state = dot(self.state_transition, self.current_state)
        self.covariance = dot(self.state_transition, dot(self.covariance, self.state_transition.T)) + self.measurement_noise

    def update(self, measurement):
        Ck = dot(self.measuring_matrix, self.current_state)
        IS = dot(self.measuring_matrix, dot(self.covariance, self.measuring_matrix.T)) + self.process_noise_convairance
        K = dot(self.covariance, dot(self.measuring_matrix.T, inv(IS)))
        self.current_state = self.current_state + dot(K, (measurement - Ck))
        self.covariance = (np.identity(4) - K * self.measuring_matrix) * self.covariance * (np.identity(4) - K * self.measuring_matrix).T

    def get_prediction(self, measurement):
        self.projects()
        self.update()
        return measurement - dot(self.measuring_matrix, self.current_state)
