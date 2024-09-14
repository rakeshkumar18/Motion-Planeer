"""
Authors : Rakesh Kumar and Kailash Nagarajan 
Date : 12-08-2022
Iterative Linear Quadratic Regulator

This file contains the iLQR Controller loop.
"""

import pdb
import matplotlib.pyplot as plt
import numpy as np
# from Dynamics import discrete_dynamics, rollout, derivative_dynamics
from Dynamics import KinematicBicycleModel
from config import GlobalConfig
from objective_func import derivative_stage_cost, derivative_final_cost, total_cost
import time
config = GlobalConfig()

def rollout(x_0, u_traj):
    '''
    This function roll's out the trajectory given the control values and initial value of the state
    Inputs : 
            x_0 - state vector - np.arrray - shape - [NX=4, ] - float64
            u_traj - control trajectory vector - np.array - shape - [Time_steps, NU] - float64
    Outputs :
            x_traj - rolled out state trajectory vector - np.arrray - shape - [Time_steps, NX] - float64
    '''
    

    x_traj = np.zeros((config.TIME_STEPS, x_0.shape[0]))
    x_traj[0, :] = x_0

    for i in range(0, x_traj.shape[0]-1):
        dynamics = KinematicBicycleModel(x_traj[i,:],u_traj[i,:])
        x_traj[i+1, :] = dynamics.discrete_dynamics()

    return x_traj

    


def forward_pass(x_traj, u_traj, k_traj, K_traj):
    
    x_traj_new = np.empty_like(x_traj)
    u_traj_new = np.empty_like(u_traj)
    

    x_traj_new[0] = x_traj[0]

    for i in range(0, config.TIME_STEPS-1):

        u_next = u_traj[i] + config.ALPHA*k_traj[i] + K_traj[i]@(x_traj_new[i] - x_traj[i])
        
        u_traj_new[i] = u_next 

        dynamics = KinematicBicycleModel(x_traj_new[i], u_traj_new[i])
        x_next = dynamics.discrete_dynamics()
        x_traj_new[i+1] = x_next
    
    return x_traj_new, u_traj_new

def backward_pass(x_traj, u_traj, regu, ref_traj):
    
    k_traj = np.empty_like(u_traj)
    K_traj = np.empty((config.TIME_STEPS-1, config.NU, config.NX))

    V_x, V_xx = derivative_final_cost(x_traj[-1], ref_traj[-1])
    
    expected_cost_redu = 0.0

    for i in range(0, config.TIME_STEPS-1):
        n = config.TIME_STEPS-2-i
        l_x, l_u, l_xx, l_ux, l_uu = derivative_stage_cost(x_traj[n], u_traj[n], ref_traj[n])

        dynamics = KinematicBicycleModel(x_traj[n], u_traj[n])
        f_x, f_u = dynamics.derivative_dynamics()
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)

        Q_uu_regu = Q_uu + np.eye(config.NU)*regu
    
        k, K = gain_terms(Q_uu_regu, Q_u, Q_ux)
        
        k_traj[n, :] = k
        K_traj[n, :, :] = K

        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, k, K)
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)

    return k_traj, K_traj, expected_cost_redu



def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    '''
    This function calculates the Q values.
    Inputs : 
            l_x - jacobian of stage cost w.r.t the state - jax.ndarray - shape - [4, ] - float32
            l_u - jacobian of stage cost w.r.t the control - jax.ndarray - shape - [2, ] - float32
            l_xx - hessian of stage cost w.r.t the state - jax.ndarray - shape - [4, 4] - float32
            l_uu - hessian of stage cost w.t.t the control - jax.ndarray - shape - [2, 2] - float32
            l_ux - hessian of stage cost w.r.t the state & control - jax.ndarray - shape [4, 4] - float32
            f_x - jacobian of dynamics w.r.t the state vector - jax.ndarray - shape - [4, ] - float32
            f_u - jacobian of dynamics w.r.t the control vector - jax.ndarray - shape - [4, ] - float32
            V_x -  the value function w.r.t the state vector - jax.ndarray - shape - [4, ] - float32
            V_xx - the value function w.r.t the state vector - jax.ndarray - shape - [4, 4] -float32
    Outputs :
            Q_x - the Q_value w.r.t the state vector - jax.ndarray - shape - [4, ] - float32
            Q_u - the Q_value w.r.t the control vector - jax.ndarray - shape - [2, ] - float32
            Q_xx - the Q_value w.r.t the state vector, state vector - jax.ndarray - shape - [4, 4] - float32
            Q_ux - the Q_value w.r.t the control vector, state vector - jax.ndarray - shape - [4, 4] - float32
            Q_uu - the Q_value w.r.t the control vector, control vector - jax.ndarray - shape - [2, 2] - float32
    '''
 
    Q_x = l_x + f_x.T@V_x
    Q_u = l_u + f_u.T@V_x
    Q_xx = l_xx + f_x.T@V_xx@f_x 
    Q_uu = l_uu + f_u.T@(V_xx)@f_u 
    Q_ux = l_ux + f_u.T@(V_xx)@f_x  

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, k, K):
    '''
    This function calculates the Value function terms.
    Inputs : 
            Q_x - the Q_value w.r.t the state vector - jax.ndarray - shape - [4, ] - float32
            Q_u - the Q_value w.r.t the control vector - jax.ndarray - shape - [2, ] - float32
            Q_xx - the Q_value w.r.t the state vector, state vector - jax.ndarray - shape - [4, 4] - float32
            Q_ux - the Q_value w.r.t the control vector, state vector - jax.ndarray - shape - [4, 4] - float32
            Q_uu - the Q_value w.r.t the control vector, control vector - jax.ndarray - shape - [2, 2] - float32
    Outputs :
            V_x -  the value function w.r.t the state vector - jax.ndarray - shape - [4, ] - float32
            V_xx - the value function w.r.t the state vector - jax.ndarray - shape - [4, 4] -float32
    '''
    
    V_x = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
    V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K
   
    return V_x, V_xx


def gain_terms(Q_uu, Q_u, Q_ux):

    '''
    This function calculates the gain matrices
    Inputs : 
            Q_u - the Q_value w.r.t the control vector - jax.ndarray - shape - [2, ] - float32
            Q_ux - the Q_value w.r.t the control vector, state vector - jax.ndarray - shape - [4, 4] - float32
            Q_uu - the Q_value w.r.t the control vector, control vector - jax.ndarray - shape - [2, 2] - float32
    Outputs :
            k - the gain matrix w.r.t the Q value of the control vector - jax.ndarray - shape - [ ] - float32
            K - the gain matrix w.r.t the Q value of the control, state vector - jax.ndarray - shape - [ ] - float32
    '''
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = -Q_uu_inv@Q_u 
    K = -Q_uu_inv@Q_ux 
    
    return k, K

def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T@k - 0.5 * k.T@Q_uu@k

def run_ilqr(x_0, u_traj, ref_traj):
    '''
    This function runs the main ilqr loop. 
    Inputs : 
            x_0 - the initial state vector - jax.ndarray - shape - [NX, ] - float32
            u_traj - the control vector - jax.ndarray - shape - [n , 2] - float32
            ref_traj - the reference trajectory - jax.ndarray - shape - [n, NX] - float32
    Outputs : 
            x_traj - the state vector - jax.ndarray - shape - [n, NX] - float32
            u_traj - the control vector - jax.ndarray - shape - [NX, 2] - float32
            cost_trace - the cost vector for the full trajectory - shape - [max_iter, ] - float32
    '''
    regu = config.regu_init
    
    x_traj = rollout(x_0, u_traj)
    
    cost_trace = np.zeros(config.max_iter+1)

    cost_trace[0] = total_cost(x_traj, u_traj, ref_traj)

    for i in range(1,config.max_iter+1):
        
        k_traj, K_traj, expected_cost_redu = backward_pass(x_traj, u_traj, regu, ref_traj)        
        x_traj_new, u_traj_new = forward_pass(x_traj, u_traj, k_traj, K_traj)
        
        total_cost_ = total_cost(x_traj_new, u_traj_new, ref_traj)

        if (cost_trace[i-1] - total_cost_ > 1e-6):
            cost_trace[i] = total_cost_
            x_traj = x_traj_new
            u_traj = u_traj_new
            regu *= config.regu_true
        else :
            cost_trace[i] = cost_trace[i-1]
            regu *=config.regu_false            
        
        max_regu = config.max_regu 
        min_regu = config.min_regu

        regu += np.maximum(min_regu - regu, 0)
        regu -= np.maximum(regu - max_regu, 0)
    
    return x_traj, u_traj, cost_trace

    