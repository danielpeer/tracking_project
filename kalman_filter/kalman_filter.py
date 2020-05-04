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
        self.covariance = dot(self.state_transition, dot(self.covariance, self.state_transition.T))

    def update(self, measurement):
        Ck = dot(self.measuring_matrix, self.current_state)
        # if np.array_equal(measurement, np.array([[-1], [-1]])):
         #   self.process_noise_covariance = init_R_maximum()
        IS = dot(self.measuring_matrix, dot(self.covariance, self.measuring_matrix.T)) + self.process_noise_covariance
        K = dot(self.covariance, dot(self.measuring_matrix.T, inv(IS)))
        a = measurement - Ck
        self.current_state = self.current_state + dot(K, (measurement - Ck))
        self.covariance = dot(np.identity(4) - dot(K, self.measuring_matrix), self.covariance) + self.measurement_noise

    def get_prior_estimate(self):
        self.projects()
        return dot(self.measuring_matrix, self.current_state)

    def get_posterior_estimate(self, measurement):
        self.update(measurement)
        return dot(self.measuring_matrix, self.current_state)

    def update_process_noise_covariance(self, center_of_mass_prediction, correlation_prediction):
        x_deviation = abs(center_of_mass_prediction[0]-correlation_prediction[0])
        y_deviation = abs(center_of_mass_prediction[1]-correlation_prediction[1])
        if(x_deviation < 5  and y_deviation < 5):
            self.process_noise_covariance = init_R_minimum()
        else :
            self.process_noise_covariance = init_R_maximum()
        #self.process_noise_covariance = np.array([[x_deviation, 0], [0, y_deviation]])


    def get_prediction(self, measurement):
        self.projects()
        self.update(measurement)
        return dot(self.measuring_matrix, self.current_state)
