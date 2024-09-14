"""
Authors : Rakesh Kumar
Date : 20-09-2022
Iterative Linear Quadratic Regulator with manual hessian and jacobian calculatin

This file contains the global configuration parameters class.
"""
import numpy as np
class GlobalConfig:

    TIME_STEPS = 10     # Horizon/ Number of waypoints used for trajectory prediction
    NU = 2                 # Number of control input ( accleration, steering angle)
    NX = 4                 # Number of state(x_pose, y_pose, v, yaw)
    max_iter = 15       # Maximum iteration of iLQR main "ruin_ilqr"
    regu_init = 1000.0      
    regu_true = 0.4
    regu_false = 7.0 
    max_regu = 10.0
    min_regu = 0.01

    ALPHA = 0.20           # Line - Search parameter
    dt = 0.1               # Control frequency, Dont touch it
    desired_velocity = 20 # Desired max velocity
    a_lat = 1.5          # Maximum lateral acceleration(m/s/s)

    ''' Geometric Parameter of the vehicle model

        LF = Distance between front axle and cg of the vehicle

        LR = Distance between rear axle and cg of the vehicle

        L = Wheel base of the car
    '''
    LF = 1.105                  # Required from the vehicle geometry
    LR = 1.738
    L = LF + LR

    ''' Tuning Parameters:

    '''
    acc_min = -3 # m/s**2
    acc_max = 3    # m/s**2

    # Tunning Weights of tracking for stage_cost
    w_ref = 4.0
    w_vel = 1.0
    w_yaw = 0.0

    # Tunning Weights of final cost

    w_ref_final = 2.0
    w_vel_final = 1.0
    w_yaw_final = 0.0

    # Tunining parameter for controls
    wacc = 0.1/acc_max
    wyawr =0.1/np.radians(30)

    # Shape tuning parameter of the barrier function
    q1 = 0.5
    q2 = 0.75

    # Steering and acceleration limit
    steer_min = -np.radians(30)
    steer_max =  np.radians(30)

    

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)