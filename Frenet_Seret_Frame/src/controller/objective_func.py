import contextlib
from re import T
from config import GlobalConfig
import numpy as np
config = GlobalConfig()
import pdb

def total_cost(x_traj, u_traj, ref_traj):

        # Normalizing the input vectors to the cost functions
        # x_traj = x_traj/(np.linalg.norm(x_traj))
        # u_traj = u_traj/(np.linalg.norm(u_traj))
        # ref_traj= ref_traj/(np.linalg.norm(ref_traj))

        '''
        This function returns the total cost for the trajectory
        Inputs : 
                x_traj - trajectory states - jax.ndarray - shape - [n=TIME_STEPS, NX] - float64
                u_traj - trajectory control - jax.ndarray - shape - [n, NU] - float64            
                ref_traj - reference trajectory - jax.ndarray - shape -[n, NU] - float64
        Outputs :
                total - total cost of the trajectory - float64 value.
        '''

        total = 0.0
        for i in range(0, config.TIME_STEPS-1):

                total = total + stage_cost(x_traj[i,:], u_traj[i,:], ref_traj[i,:]) 
                
        total += final_cost(x_traj[-1], ref_traj)

        return total 

def stage_cost(x, u, ref_traj):
        

        '''
        This function calculates sqaure of the L-2 distance between the current pos and the reference pos.
        Inputs :
                x - current state - jax.ndarray - shape - [NX=3, ] - float64
                ref_traj - reference pos - jax.ndarray - shape - [NX, ] - float64
        Outputs :
                distance - calculated L-2 distance - float64 value.
        ''' 
        global Q, R 

        w_ref = config.w_ref ; w_vel = config.w_vel ; w_yaw = config.w_yaw   # Tuning Weights
        q = np.array([w_ref, w_ref,w_vel, w_yaw])  # Diagonal term for for state weights
        Q = np.diag(q) # State weight matrix
        
        wacc = config.wacc
        wyawr = config.wyawr

        r = np.array([wacc,wyawr])
        R = np.diag(r)   # Control Weight Matrix

        # Constraint cost for steering to limit and accelaration control
        P1 = np.array([[1],[0]])         # Vector to choose acceleration                  
        P2 = np.array([[0],[1]])         # Vector to choose steering limit

        # Shape tuning parameter of the barrier function
        q1 = config.q1
        q2 = config.q2 

        steer_min = config.steer_min
        steer_max = config.steer_max

        # Yawrate Barrier Max
        c1 = (u.T@ P2 - steer_max)   # Constraint function on steering angle 
        b_1, b_dot_1, b_ddot_1 = barrier_function(q1, q2, c1, P2)

        # Yawrate Barrier Min
        c2 = (steer_min - u.T @ P2)  # Constraint function on steering angle 
        b_2, b_dot_2, b_ddot_2 = barrier_function(q1, q2, c2, -P2)

        # Accelartion Barrier Max
        acc_min = config.acc_min
        acc_max = config.acc_max

        c3 = (u.T@ P1 - acc_max)   # Constraint function on steering angle 0
        b_3, b_dot_3, b_ddot_3 = barrier_function(q1, q2, c3, P1)

        # Accelartion Barrier min
        c4 = (acc_min - u.T @ P1)  # Constraint function on steering angle 
        b_4, b_dot_4, b_ddot_4 = barrier_function(q1, q2, c4, -P1)

        #     pdb.set_trace()

        total_stage_cost = 0.5*(x.T-ref_traj.T).T @ Q @ (x.T-ref_traj.T) \
                + 0.5*(u  ) @ R @ (u ).T \
                        + (u  ).T @ b_ddot_1 @ u + (u  ) @ b_dot_1 + b_1 \
                        + (u  ) @ b_ddot_2 @ u + (u  ) @ b_dot_2 + b_2 \
                        + (u  ).T @ b_ddot_3 @ u + (u  ) @ b_dot_3 + b_3 \
                        + (u  ) @ b_ddot_4 @ u + (u  ) @ b_dot_4 + b_4
                

                
        # print("state_cost", 0.5*(x.T-ref_traj.T).T @ Q @ (x.T-ref_traj.T))
        # print("control_cost",0.5*(u  ) @ R @ (u ).T)
        return total_stage_cost


def final_cost(x, ref_traj):

        '''
        This function returns the cost at the goal point of the trajectory
        Inputs : 
                x - current state - jax.ndarray - shape - [4, ] - float64
                ref_traj - reference pos - jax.ndarray - shape -[n, 4] - float64
        Outputs :
                terminal_pos_cost - final cost at the goal point - float64 value.
        '''
        global QN

        w_ref = config.w_ref_final; w_vel = config.w_vel_final; w_yaw = config.w_yaw_final
        q = np.array([w_ref, w_ref,w_vel,w_yaw])  # Diagonal term for for state weights
        QN = np.diag(q)    #Final State weight matrix

        terminal_pos_cost = 0.5 * (x.T-ref_traj[-1].T).T @ QN @ (x.T-ref_traj[-1].T)
        
        return terminal_pos_cost
    

def derivative_stage_cost(x, u, ref_traj):
        # Normalizing the input vectors to the cost functions
        # x = x/(np.linalg.norm(x))
        # u = u/(np.linalg.norm(u))
        # ref_traj= ref_traj/(np.linalg.norm(ref_traj))
  
        '''
        This function returns the Jacobians and Hessians of the stage cost.
        Inputs :
                x - current state - jax.ndarray - shape - [NX, ] - float64
                u - current control - jax.ndarray - shape - [NU, ] - float64            
                ref_traj - reference pos - jax.ndarray - shape -[NU, ] - float64
                
        Outputs :
                l_x - jacobian of stage cost w.r.t the state - jax.ndarray - shape - [NX, ] - float64
                l_u - jacobian of stage cost w.r.t the control - jax.ndarray - shape - [NU, ] - float64
                l_xx - hessian of stage cost w.r.t the state - jax.ndarray - shape - [NX, NX] - float64
                l_uu - hessian of stage cost w.t.t the control - jax.ndarray - shape - [NU, NU] - float64
                l_ux - hessian of stage cost w.r.t the state & control - jax.ndarray - shape [NU, NX] - float64
        '''
        # Constraint cost for steering_limit
        
        q1 = config.q1; q2 = config.q2 # Shape tuning parameter of the barrier function
        steer_min = config.steer_min
        steer_max = config.steer_max

        P1 = np.array([[1],[0]])
        P2 = np.array([[0],[1]])
        # steering_angle Barrier Max
        c = (u.T@P2 - steer_max)
        b_1, b_dot_1, b_ddot_1 = barrier_function(q1, q2, c, P2)

        # steering_angle Barrier Min
        c = (steer_min - np.matmul(u.T, P2))
        b_2, b_dot_2, b_ddot_2 = barrier_function(q1, q2, c, -P2)

        # Accelartion Barrier Max
        acc_min = config.acc_min
        acc_max = config.acc_max

        c3 = (u.T@ P1 - acc_max)   # Constraint function on steering angle 
        b_3, b_dot_3, b_ddot_3 = barrier_function(q1, q2, c3, P1)

        # Accelartion Barrier min
        c4 = (acc_min - u.T @ P1)  # Constraint function on steering angle 
        b_4, b_dot_4, b_ddot_4 = barrier_function(q1, q2, c4, -P1)


        l_x = Q @ (x.T-ref_traj.T)
        l_u = R @ (u).T  + b_ddot_1 @ (u  ) + b_ddot_2 @ (u  )\
                + b_ddot_3 @ (u  ) + b_ddot_4 @ (u  )\
                + b_dot_1.reshape(2,) + b_dot_2.reshape(2,)\
                + b_dot_3.reshape(2,) + b_dot_4.reshape(2,)


        l_xx = Q
        l_uu = R + b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4
        l_ux = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]) 
        
        return l_x, l_u, l_xx, l_ux, l_uu
    

def derivative_final_cost(x, goal_point):

        # # Normalizing the input vectors to the cost functions
        # x = x/(np.linalg.norm(x))
        # goal_point = goal_point/(np.linalg.norm(goal_point))
        
        '''
        This function returns the Jacobian and Hessian of the terminal cost.
        Inputs :
                x - current state - jax.ndarray - shape - [NX, ] - float64        
                goal_point - final pos of the trajectory - jax.ndarray - shape -[NU, ] - float64
                
        Outputs :
                l_final_x - jacobian of final cost w.r.t the state - jax.ndarray - shape - [NX, ] - float64
                l_final_xx - hessian of final cost w.r.t the state - jax.ndarray - shape - [NX, NX] - float64
        '''

        l_final_x = QN @ (x.T-goal_point.T)
        l_final_xx = QN

        return l_final_x, l_final_xx

def barrier_function(q1, q2, c, c_dot):

        b = q1*np.exp(q2*c)
	
        b_dot = q1*q2*np.exp(q2*c)*c_dot
		
        b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)
        
        return b, b_dot, b_ddot








































# x_0 = np.array([-0.5, -0.5, 1.0, 0.5])
# u_traj = np.random.randn(config.TIME_STEPS-1, config.NU)
# # A, B = derivative_dynamics(x_0, u)

# x_traj = rollout(x_0, u_traj)
# t_traj  = np.arange(0, 10, config.dt)
# v_x = config.desired_velocity


# x_ref, y_ref, yaw_ref, v_ref = sine_curve_trajectory(t_traj, v_x)
# ref_traj = np.array([x_ref, y_ref,v_ref, np.zeros(x_ref.shape)]).T

# Total_cost = total_cost(x_traj, u_traj, ref_traj)
# print(Total_cost)