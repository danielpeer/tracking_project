from numpy import dot
from numpy.linalg import inv
from kalman_filter.kalman_utilities import *


class KalmanFilter:
    def __init__(self, target_info, frames_per_second):
        initial_location = target_info.current_pos
        init_global_variables(frames_per_second)

        # the mean state estimate of the previous step
        self.current_state = init_state_vector(initial_location)

        # the transition matrix
        self.state_transition = init_state_transition(frames_per_second)

        # the state covariance of previous step
        self.covariance = init_covariance_matrix()

        # the process noise covariance matrix
        self.process_noise_covariance = init_minimum_process_noise_covariance()

        # the measuring matrix
        self.measuring_matrix = init_measuring_matrix()

        # the measurement noise covariance matrix
        self.measurement_noise = init_minimum_measurement_noise()

    def _projects(self):
        self.current_state = dot(self.state_transition, self.current_state)
        self.covariance = dot(self.state_transition, dot(self.covariance, self.state_transition.T)) + self.measurement_noise

    def _update(self, measurement):
        Ck = dot(self.measuring_matrix, self.current_state)
        IS = dot(self.measuring_matrix, dot(self.covariance, self.measuring_matrix.T)) + self.process_noise_covariance
        K = dot(self.covariance, dot(self.measuring_matrix.T, inv(IS)))
        a = measurement - Ck
        self.current_state = self.current_state + dot(K, (measurement - Ck))
        self.covariance = dot(np.identity(4) - dot(K, self.measuring_matrix), self.covariance)

    def base_kalman_prior_prediction(self):
        """
            increase process noise covariance to take into account only Kalman's prior prediction
            OVERLAP/CONCEALMENT STATE
        """
        self.process_noise_covariance = init_maximum_process_noise_covariance()
        self.measurement_noise = init_minimum_measurement_noise()

    def base_measurement(self):
        """
            decrease process noise covariance to consider only the measurement
        """
        self.process_noise_covariance = init_minimum_process_noise_covariance()
        self.measurement_noise = init_maximum_measurement_noise()

    def get_prediction(self, measurement):
        """
            get Kalman's prediction
        """
        self._projects()
        self._update(measurement)
        return dot(self.measuring_matrix, self.current_state).astype(int)
