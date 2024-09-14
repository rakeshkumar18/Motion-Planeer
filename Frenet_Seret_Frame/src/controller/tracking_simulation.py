import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from config import GlobalConfig
from ilqr_maths_main import run_ilqr
import pdb
config_ = GlobalConfig()

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
utils_path = project_root / 'utils'
sys.path.insert(0, str(utils_path))

from cubic_spline_planner import CubicSpline2D

# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent

# Navigate up to the project root
project_root = script_dir.parent

# Build the path to the 'data' directory and the CSV file
data_path = project_root / 'data' / 'eight_shaped_road.csv'

# Debugging: Print paths to verify
# print(f"Script directory: {script_dir}")
# print(f"Project root: {project_root}")
# print(f"Data path: {data_path}")

# Check if the file exists
if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

def sigmoid(z):
    return 1/(1+ np.exp(-z))
import pandas as pd
def main():
    # Load waypoints from CSV file
    df = pd.read_csv(data_path)
    x = df['x'].values
    y = df['y'].values
    
    N = 600  # Total number of waypoints from the cubic spline planner
    sp = CubicSpline2D(x, y)
    Total_Arc_length = sp.s[-1]
    Piecewise_arc_length = Total_Arc_length / N
    s = np.arange(0, sp.s[-1], Piecewise_arc_length)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    x_ref = np.array(rx)
    y_ref = np.array(ry)
    yaw_ref = np.array(ryaw)
    kappa = np.array(rk)

    v_ref = np.round(np.sqrt(config_.a_lat / (abs(kappa) + 0.01)), 2)
    
    # Reference trajectory and control
    ref_traj = np.array([x_ref, y_ref, v_ref, yaw_ref]).T

    x_0 = np.array([x_ref[0], y_ref[0], v_ref[0], yaw_ref[0]])
    states = np.array([[x_ref[0], y_ref[0], 0.0, 0.0]])
    u_traj = np.ones([config_.TIME_STEPS - 1, config_.NU]) * 0.0000
   
    controls = np.array([[u_traj[0, 0], u_traj[0, 1]]])

    t = 0
    tf = 200
    t_ = []

    # Initialize lists to store trajectory data
    x_traj, y_traj = [], []

    # Create a directory to save frames
    import os
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # Simulation loop
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(x_ref, y_ref, color='gray', linewidth=5, label='Reference Trajectory')

    frame_count = 0

    while t <= tf:
        target_idx, _ = calc_target_index(x_0[0], x_0[1], x_0[-1], x_ref, y_ref)
        if target_idx >= len(x_ref) - config_.TIME_STEPS:
            break

        x_, u_, cost_trace = run_ilqr(x_0, u_traj, ref_traj[target_idx:target_idx + config_.TIME_STEPS])
        
        x_0 = x_[1]  # Update horizon initialization
        u_traj = u_  # Update control input
        
        controls = np.append(controls, np.array([u_[0, :]]), axis=0)
        states = np.append(states, np.array([x_[0, :]]), axis=0)

        t_.append(t)
        t += 0.1
        x_traj.append(x_0[0])
        y_traj.append(x_0[1])

        ax.clear()
        ax.plot(x_ref, y_ref, color='gray', linewidth=5, label='Reference Trajectory')
        ax.plot(x_[:, 0], x_[:, 1], 'r', linewidth=1, label='Receding Horizon Prediction')
        ax.plot(x_[0, 0], x_[0, 1], 'o:g', markersize=8, label='Tracked Trajectory')
        ax.set_title(f'Waypoints Tracking: Speed [km/h]: {x_0[2] * 3.6:.2f}', fontsize=14)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True)
        ax.legend(loc='best')
        plt.pause(0.001)

        # Save the frame
        plt.savefig(f'frames/frame_{frame_count:03d}.png')
        frame_count += 1

    # Save the last position
    x_traj.append(x_0[0])
    y_traj.append(x_0[1])
    plt.close()

    # Create a GIF from the saved frames
    data_path_sim = project_root / 'data' / 'simulation.gif'
    with imageio.get_writer(data_path_sim, mode='I', duration=0.1) as writer:
        for i in range(frame_count):
            image = imageio.imread(f'frames/frame_{i:03d}.png')
            writer.append_data(image)


    # Speed tracking over time
    plt.figure()
    plt.plot(t_, v_ref[0:len(t_)], 'k', linewidth=2, label='Reference Speed')
    plt.plot(t_, states[0:len(t_), 2], 'r', linewidth=2, label='Tracked Speed')
    plt.title('Speed Tracking')
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    

def calc_target_index(x, y, yaw, cx, cy):
    """

    Args:
        - state
        - cx: in metere
        - cy: in meter

    Compute index in the trajectory list of the target.
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    L = config_.L
   # Calc front axle position
    fx = x + L * np.cos(yaw) # yaw is in radians
    fy = y + L * np.sin(yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = [np.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
    closest_error = min(d)
    target_idx = d.index(closest_error)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(yaw + np.pi / 2),
                      - np.sin(yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle
       
main()