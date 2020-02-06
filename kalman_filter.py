'''
Created on Feb 5, 2020
@author: Oyindamola Omotuyi

Kalman Filter Algorithm for a one dimensional target tracking problem

'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import multi_dot



class KalmanFilter(object):
    def __init__(self, stateTransitionMatrix, observationMatrix, control = 0, controlInputMatrix = 0):     
        self.import_data()
        
        if(stateTransitionMatrix is None):
            raise ValueError("State Transition Matrix not set!")
            
        if(observationMatrix is None):
            raise ValueError("Observation Matrix not set!")
            
        self.STATE_SIZE = stateTransitionMatrix.shape[0]
        self.MEASUREMENT_SIZE = observationMatrix.shape[0]
        
        #STATE PARAMETERS
        self.state = np.zeros(self.STATE_SIZE) #X
        self.stateTransitionMatrix = stateTransitionMatrix #F
        self.estimateCovariance = np.identity(self.STATE_SIZE) #P 
        self.processNoiseCovariance = 0.1*np.identity(self.STATE_SIZE) #Q                                       
        
        
        #MEASUREMENT PARAMETERS
        self.observationMatrix = observationMatrix #H
        self.observationCovariance = np.identity(self.MEASUREMENT_SIZE)
         
        #CONTROL PARAMETERS
        self.controlInputMatrix = controlInputMatrix
        self.control = control

        
    def import_data(self): 
        # Imports true data and sensor readings from CSV files.
        self.true_data= np.genfromtxt('Data/true_data.csv', delimiter=',')
        self.sen_pos_data = np.genfromtxt('Data/sen_pos_data.csv', delimiter=',')
        self.sen_acc_data = np.genfromtxt('Data/sen_acc_data.csv', delimiter=',')
 

    def predict(self): 
        #(1) Predict state with X = AX + BU                  
        self.state =  np.dot(self.stateTransitionMatrix, self.state) + np.dot(self.controlInputMatrix, self.control)  
        
        #(2) Estimate state Covariance
        self.estimateCovariance = multi_dot([self.stateTransitionMatrix, self.estimateCovariance, 
                                             np.transpose(self.stateTransitionMatrix)]) + self.processNoiseCovariance
            

    def update(self):

        self.innovation = self.measurement - np.dot(self.observationMatrix,  self.state) # y = z - Hx

        self.innovation_covariance = multi_dot([self.observationMatrix,
                                            self.estimateCovariance,
                                            np.transpose(self.observationMatrix)]) + self.observationCovariance #S

        self.kalman_gain = multi_dot([self.estimateCovariance,
                                np.transpose(self.observationMatrix),
                                inv(self.innovation_covariance)]) #K
     
        self.state = self.state + np.dot(self.kalman_gain, self.innovation) # x = x + K*y

        self.identity = np.identity(self.STATE_SIZE)

        first_term = self.identity - np.dot(self.kalman_gain,self.observationMatrix) # (I-K*H)

        #Using the Joseph form: (I - KH)P(I - KH)' + KRK'
        self.estimateCovariance = multi_dot([first_term,
                                            self.estimateCovariance,
                                            np.transpose(first_term)]) + multi_dot([self.kalman_gain,
                                                                                    self.observationCovariance,
                                                                                    np.transpose(self.kalman_gain)])


def targetTrackingProblem():
    delta = 0.1 
    stateTransitionMatrix = np.array([[1,    delta,  0.5*(delta**2)],
                                       [0,    1,      delta],
                                        [0,    0,       1]]) #F
                                         
                      
    processNoiseCovariance = 0.1**2 *np.array([[(delta**4)/4,    (delta**3)/2,      (delta**2)/2],
                                                [(delta**3)/2,    2*(delta**3),       delta**2],
                                                [(delta**2)/2,    delta**2,           delta**2]]) #Q 
    
    T = 20
    estimated_state_pos = np.ones((200,3))
    estimated_state_acc = np.ones((200,3))
    time_steps = np.linspace(0,T,200)
    
    #Kalman Filter Object for Position Measurements
    observationMatrix_pos= np.array([[1, 0, 0]]) #H
    filter_with_pos_ = KalmanFilter(stateTransitionMatrix, observationMatrix_pos)
    filter_with_pos_.processNoiseCovariance = processNoiseCovariance
    filter_with_pos_.observationCovariance = np.array([[0.9**2]]) #R 
    
    #Kalman Filter Object for Acceleration Measurements
    observationMatrix_acc= np.array([[0, 0, 1]]) #H
    filter_with_acc_ = KalmanFilter(stateTransitionMatrix, observationMatrix_acc)
    filter_with_acc_.processNoiseCovariance = processNoiseCovariance
    filter_with_acc_.observationCovariance = np.array([[1.5**2]]) #R 
    
    prev_time = 0
    
    for i in range(len(filter_with_pos_.true_data)):
        current_time = time_steps[i]
        delta = current_time - prev_time
        prev_time = current_time
        
        filter_with_pos_.predict()
        filter_with_pos_.measurement = np.array([filter_with_pos_.sen_pos_data[i]])
        filter_with_pos_.update()
        estimated_state_pos[i] = filter_with_pos_.state
        
        
        filter_with_acc_.predict()
        filter_with_acc_.measurement = np.array([filter_with_acc_.sen_acc_data[i]])
        filter_with_acc_.update()    
        estimated_state_acc[i] =filter_with_acc_.state

    #Estimated States using Position Measurements
    est_pos_pos = estimated_state_pos[:,0]
    est_vel_pos = estimated_state_pos[:,1]
    est_acc_pos = estimated_state_pos[:,2]
    
    #Estimated States using Acceleration Measurements
    est_pos_acc = estimated_state_acc[:,0]
    est_vel_acc = estimated_state_acc[:,1]
    est_acc_acc = estimated_state_acc[:,2]
    
    ##PLOTS
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Kalman Filter')
    
    #Position Data with Position Sensor Measurements
    ax1.plot(time_steps, filter_with_pos_.true_data, '--r', label="True")
    ax1.plot(time_steps, est_pos_pos, '--b', label="Estimated")
    ax1.grid(True)
    ax1.set(xlabel='Time (secs)', ylabel="Position (m)")
    ax1.set_title('True Position vs Estimated Position for Position Measurements')
    ax1.legend()
    
    #Position Data with Acceleration Sensor Measurements
    ax2.plot(time_steps, filter_with_pos_.true_data, '--r', label="True")
    ax2.plot(time_steps, est_pos_acc, '--b', label="Estimated")
    ax2.grid(True)
    ax2.set(xlabel='Time (secs)', ylabel="Position (m)")
    ax2.set_title('True Position vs Estimated Position for Acceleration Measurements')
    ax2.legend()
    
    #Estimated Velocity 
    ax3.plot(time_steps, est_vel_pos, '-k', label="Pos_Data")
    ax3.plot(time_steps, est_vel_acc, '-r', label="Acc_Data")
    ax3.grid(True)
    ax3.set(xlabel='Time (secs)', ylabel="Velocity (m/s)")
    ax3.set_title('Estimated Velocity')
    ax3.legend()
    
    #Estimated Acceleration  
    ax4.plot(time_steps, est_acc_pos, '-k', label="Pos_Data")
    ax4.plot(time_steps, est_acc_acc, '-r', label="Acc_Data")
    ax4.grid(True)
    ax4.set(xlabel='Time (secs)', ylabel="Acceleration (m/s^2)")
    ax4.set_title('Estimated Acceleration')
    ax4.legend()

                                    
    
if __name__ == '__main__':
    targetTrackingProblem()
    
