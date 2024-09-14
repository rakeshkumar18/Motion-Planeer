import numpy as np
import matplotlib.pyplot as plt
from config import GlobalConfig
np.set_printoptions(precision=3)
config = GlobalConfig()
import pdb

class KinematicBicycleModel:

    '''state = [x_pose, y_pose, v, yaw]  # shape = (NX, )
       control_input = [acceleration, steering_angle] 
        x[0] = x_pose

        x[1] = y_pose

        x[2] = v

        x[3] = yaw        
        
        u[0] = acceleration

        u[1] = steering_angle
        '''
    def __init__(self, x, u):
        self.x = x
        self.u = u

    def discrete_dynamics(self):
        acceleration = self.u[0]
        steering_angle = self.u[1]

        x_d_1 = np.array([self.x[0] + np.cos(self.x[3]) * (self.x[2] * config.dt + 0.5 * acceleration * config.dt **2),
                        
                        self.x[1] + np.sin(self.x[3]) * (self.x[2] * config.dt + 0.5 * acceleration*config.dt**2),
            
                        self.x[2] +  acceleration * config.dt,
                    
                        self.x[3] + self.x[2]/config.L*np.tan(steering_angle) * config.dt
                    ])
        
        x_d = x_d_1 #+ np.random.normal(0,0.05,4).T                   
        return x_d
        

    def derivative_dynamics(self):
        '''
    This function calculates the jacobian of the dynamics, to linearize it. Jacobians for the dynamics have been hand calculated
    Inputs :
           x - current state - np.arrray - shape - [NX, ] - float64
           u - current control - np.arrray - shape - [NU, ] - float64
    Outputs :
            f_x - jacobian of dynamics w.r.t the state vector - np.arrray - shape - [NX, NX] - float64
            f_u - jacobian of dynamics w.r.t the control vector - np.arrray - shape - [NX, NU] - float64 
    '''
   
        acceleration = self.u[0]
        steering_angle = self.u[1]

        f_x = np.array([[1, 0, np.cos(self.x[3])*config.dt, -(self.x[2]*config.dt + 0.5 * acceleration*config.dt**2)*np.sin(self.x[3])],
                        
                        [0, 1, np.sin(self.x[3])*config.dt,  (self.x[2]*config.dt + 0.5 * acceleration*config.dt**2)*np.cos(self.x[3])],
                        
                        [0, 0,             1,                                         0],
                        
                        [0, 0,      1/config.L*np.tan(steering_angle)* config.dt,     1]
                        
                        ])
        
        f_u = np.array([[0.5 * config.dt **2 * np.cos(self.x[3]), 0],
                        
                        [0.5 * config.dt **2 * np.sin(self.x[3]), 0],
                        
                        [config.dt, 0],
                        
                        [0, self.x[2]/config.L * (1/np.cos(steering_angle))**2 *config.dt]
                        
                        ])
        return f_x, f_u


































# Tests for the Dynamics.py
# x_0 = np.array([-0.5, -0.5, 1.0, 0.0])
# u_traj = np.random.randn(config.TIME_STEPS-1, config.NU)
# # A, B = derivative_dynamics(x_0, u)
# x_traj = rollout(x_0, u_traj)
# print(x_traj)

# u = np.array([2.0,0.5])
# f_x, f_u = derivative_dynamics(x_0, u)
# print("f_x\n", f_x)
# print("f_u\n", f_u)

#Sine curve trajectory
# t_traj  = np.arange(0, 10, config.dt)
# v_x = config.desired_velocity


# x_ref, y_ref, yaw_ref, v_ref = sine_curve_trajectory(t_traj, v_x)
# ref_traj = np.array([x_ref, y_ref,v_ref, np.zeros(x_ref.shape)]).T

# Total_cost = total_cost(x_traj, u_traj, ref_traj)
# pdb.set_trace()


# u = np.array([0.0,0.2])
# x_array = []
# y_array = []
# t_ = 0
# while t_<=100:
#     x_sol = discrete_dynamics(x_0, u)
    
#     x_0 = x_sol
#     # print(x_0)
   
#     x_array.append(x_0[0])
#     y_array.append(x_0[1])
#     print(t_)
#     t_+=0.1
# plt.figure()
# plt.plot(x_array, y_array)
# plt.show()