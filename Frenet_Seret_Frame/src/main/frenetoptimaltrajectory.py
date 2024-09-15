import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO
import copy
import math

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
utils_path = project_root / 'utils'
sys.path.insert(0, str(utils_path))

from cubic_spline_planner import CubicSpline2D
from quintic_quartic_polynomials_planner import quintic_polynomial, quartic_polynomial


# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent

# Navigate up to the project root
project_root = script_dir.parent

# Build the path to the 'data' directory and the CSV file
data_path = project_root / 'data' / 'eight_shaped_road.csv'
animation_data_path = project_root / 'data' / 'animation.gif'

# Debugging: Print paths to verify
# print(f"Script directory: {script_dir}")
# print(f"Project root: {project_root}")
# print(f"Data path: {data_path}")

# Check if the file exists
if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

# way points from data folder eight_shaped_road.csv
df = pd.read_csv(data_path)

# Check if the file exists
if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

# Parameter
MAX_SPEED = 25.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.5  # maximum acceleration [m/ss]
MAX_CURVATURE = 10.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 4  # maximum road width [m]
D_ROAD_W = 0.5  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAXT = 4.0  # max prediction time [m]
MINT = 3.0  # min prediction time [m]
TARGET_SPEED = 20.0 / 3.6  # target speed [m/s]
D_T_S = 2.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1 # sampling number of target speed
ROBOT_RADIUS = 3.0  # robot radius [m]

# cost weights

"""
KJ: Cost weights of jerk
KT: Cost weights of time
KD: Cost weights of square of derivative of lateral direction
KLAT: Cost weights of lateral direction
KLON: Cost weights of longitudinal direction
"""
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

class Frenet_path:

    def __init__(self):

        """
        t : time
        d : lateral distance
        d_d : lateral velocity
        d_dd : lateral acceleration
        d_ddd : lateral jerk
        s : longitudinal distance
        s_d : longitudinal velocity
        s_dd : longitudinal acceleration
        s_ddd : longitudinal jerk
        cd : lateral cost
        cv : velocity cost
        cf: total cost of frenet path

        x : x position
        y : y position
        yaw : yaw angle
        ds : arc length distance
        c : curvature

        """
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    """
    Args:
        c_speed: current speed
        c_d: current lateral position
        c_d_d: current lateral speed
        c_d_dd: current lateral acceleration
        s0: current longitudinal position
    """
    # Empty list for frenet paths initialized
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH , MAX_ROAD_WIDTH, D_ROAD_W):  # _MAX_ROAD_WIDTH to MAX_ROAD_WIDTH with D_ROAD_W step

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):                           # MINT to MAXT with DT step
            fp = Frenet_path()

            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Loongitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                # import pdb; pdb.set_trace()
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1])**2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

faTrajX = []
faTrajY = []

def calc_global_paths(fplist, csp):

    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # Just for plotting
        faTrajX.append(fp.x)
        faTrajY.append(fp.y)
        
        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.sqrt(dx**2 + dy**2))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


faTrajCollisionX = []
faTrajCollisionY = []
faObCollisionX = []
faObCollisionY = []

def check_collision(fp, ob):
    #pdb.set_trace()
    for i in range(len(ob[:, 0])):
        # Calculate the distance for each trajectory point to the current object
        d = [((ix - ob[i, 0])**2 + (iy - ob[i, 1])**2)
             for (ix, iy) in zip(fp.x, fp.y)]

        # Check if any trajectory point is too close to the object using the robot radius
        collision = any([di <= ROBOT_RADIUS**2 for di in d])

        if collision:
            #plot(ft.x, ft.y, 'rx')
            faTrajCollisionX.append(fp.x)
            faTrajCollisionY.append(fp.y)
            
            #plot(ox, oy, 'yo');
            #pdb.set_trace()
            if ob[i, 0] not in faObCollisionX or ob[i, 1] not in faObCollisionY:
                faObCollisionX.append(ob[i, 0])
                faObCollisionY.append(ob[i, 1])
            
            
            return True

    return False


def check_paths(fplist, ob):

    okind = []
    for i in range(len(fplist)):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        okind.append(i)

    return [fplist[i] for i in okind]

fpplist = []

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    # import pdb; pdb.set_trace()
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)
    
    #fpplist = deepcopy(fplist)
    fpplist.extend(fplist)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath

def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

show_animation = True
#show_animation = False

wx = df['x'].values
wy = df['y'].values

ob = np.array([[21, 35]
               ,[40, 20]])
tx, ty, tyaw, tc, csp = generate_target_course(wx,wy)

# initial state
c_speed = 12.0 / 3.6  # current speed [m/s]
c_d = 0.0  # current lateral position [m]
c_d_d = 0.0  # current lateral speed [m/s]
c_d_dd = 0.0  # current latral acceleration [m/s]
s0 = 0.0  # current course position

area = 50.0  # animation area length [m]

faTx = tx
faTy = ty
faObx = ob[:, 0]
faOby = ob[:, 1]
faPathx = []
faPathy = []
faRobotx = []
faRoboty = []
faSpeed = []

f_waypoints_s = []   # logitudnal Waypoints to track
f_waypoints_d = [] # lateral Waypoints to track
f_s_d = []  # logitudnal speed to track
f_d_d = []     # lateral speed to track

# Prepare lists to collect frames for the GIF
frames = []

# Main simulation loop
for i in range(20000):
    path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

    # Update vehicle states
    s0 = path.s[1]
    c_d = path.d[1]
    c_d_d = path.d_d[1]
    c_d_dd = path.d_dd[1]
    c_speed = path.s_d[1]

    f_waypoints_s.append(s0)
    f_waypoints_d.append(c_d)
    f_s_d.append(c_speed)
    f_d_d.append(c_d_d)

    # Check for goal
    if csp.s[-1] - s0 <= 2:
        print("Goal")
        break
    
    #Append robot's final trajectory


    # Append robot's path and speed
    faPathx.append(path.x[1:])
    faPathy.append(path.y[1:])
    faRobotx.append(path.x[1])
    faRoboty.append(path.y[1])
    faSpeed.append(c_speed)

    # Plotting and capturing frames for the GIF
    if show_animation:
        plt.cla()
        plt.plot(wx, wy, color='grey', label="reference_path", linewidth=10)
        plt.plot(path.x[1], path.y[1], "vc")

        for (ix, iy) in zip(faTrajX, faTrajY):
            plt.plot(ix[1:], iy[1:], '-', color=[0.5, 0.5, 0.5])
        faTrajX = []
        faTrajY = []

        for (ix, iy) in zip(faTrajCollisionX, faTrajCollisionY):
            plt.plot(ix[1:], iy[1:], 'rx')
        faTrajCollisionX = []
        faTrajCollisionY = []

        for fp in fpplist:
            plt.plot(fp.x[1:], fp.y[1:], '-g')
        fpplist = []

        for (ix, iy) in zip(faObCollisionX, faObCollisionY):
            plt.plot(ix, iy, 'oy')
        faObCollisionX = []
        faObCollisionY = []

        plt.plot(path.x[1:], path.y[1:], "-ob")
        
        plt.xlim(-area, area)
        plt.ylim(-area, area)
        plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
        plt.grid(True)

        # Capture current frame without saving to file
        buf = BytesIO()  # Create an in-memory buffer
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer to the beginning
        frames.append(imageio.imread(buf))  # Read image from the buffer into the frames list
        buf.close()  # Close the buffer

# Save the captured frames into a GIF file
gif_filename = 'animation.gif'
# Saving animation to GIF file at data folder 
imageio.mimsave(animation_data_path, frames, duration=0.1)  # 0.1 seconds per frame

# print("GIF created:", gif_filename)
print("Finish")

#=======================================================================================================================

# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent
# print(f"Script directory: {script_dir}")

# Navigate up to the project root
project_root = script_dir.parent
# print(f"Project root: {project_root}")

# Build the path to the 'data' directory
data_dir = project_root  / 'data'
# print(f"Data directory: {data_dir}")

#extract data of the robot's trajectory f_waypoints_s, f_waypoints_d, f_s_d, f_d_d
#export to .txt file
frenet_data = {'f_waypoints_s': f_waypoints_s, 'f_waypoints_d': f_waypoints_d, 'f_s_d': f_s_d, 'f_d_d': f_d_d}
# extract frenet data to .txt file and .csv file at data folder
df = pd.DataFrame(frenet_data)

# Save the data to a .txt file
df.to_csv(data_dir / 'frenet_data.txt', index=False)
# Save the data to a .csv file
df.to_csv(data_dir / 'frenet_frame_without_obstacle.csv', index=False)