import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from pathlib import Path
import sys

# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent

# Navigate up to the project root
project_root = script_dir.parent

# Build the path to the 'data' directory and the CSV file
data_path_cart = project_root / 'data' / 'eight_shaped_road.csv'
data_path_frenet = project_root / 'data' / 'frenet_frame_without_obstacle.csv'

# Debugging: Print paths to verify
# print(f"Script directory: {script_dir}")
# print(f"Project root: {project_root}")
# print(f"Data path: {data_path}")

# Check if the file exists
if not data_path_cart.exists():
    raise FileNotFoundError(f"File not found: {data_path_cart}")

# way points from data folder eight_shaped_road.csv
df_cart = pd.read_csv(data_path_cart)
df_frenet = pd.read_csv(data_path_frenet)


class CubicSpline2D:
    def __init__(self, x, y):
        """
        Initialize the 2D cubic spline for the reference path.
        """
        self.s = self.calc_s(x, y)
        self.spline_x = CubicSpline(self.s, x)
        self.spline_y = CubicSpline(self.s, y)

    def calc_s(self, x, y):
        """
        Calculate the cumulative arc length 's' from the (x, y) waypoints.
        """
        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s = np.append([0], np.cumsum(ds))
        return s

    def calc_position(self, s):
        """
        Calculate the (x, y) position for a given arc length 's'.
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        return x, y

    def calc_yaw(self, s):
        """
        Calculate the yaw (orientation) angle at a given arc length 's'.
        """
        dx = self.spline_x(s, 1)  # First derivative of x with respect to s
        dy = self.spline_y(s, 1)  # First derivative of y with respect to s
        return np.arctan2(dy, dx)

class FrenetConverter:
    def __init__(self, reference_x, reference_y):
        """
        Initialize the FrenetConverter with a reference path.
        """
        self.spline = CubicSpline2D(reference_x, reference_y)
        self.ref_s = self.spline.s

    def cartesian_to_frenet(self, x, y):
        """
        Convert Cartesian coordinates (x, y) to Frenet coordinates (s, d).
        """
        distances = np.linalg.norm(np.column_stack((self.spline.spline_x(self.ref_s) - x, 
                                                    self.spline.spline_y(self.ref_s) - y)), axis=1)
        closest_idx = np.argmin(distances)

        # Calculate arc length (s)
        s = self.ref_s[closest_idx]

        # Calculate lateral distance (d)
        ref_x, ref_y = self.spline.calc_position(s)
        dx = x - ref_x
        dy = y - ref_y
        ref_yaw = self.spline.calc_yaw(s)
        d = dx * np.sin(ref_yaw) - dy * np.cos(ref_yaw)

        return s, d

    def frenet_to_cartesian(self, s, d):
        """
        Convert Frenet coordinates (s, d) to Cartesian coordinates (x, y).
        """
        ref_x, ref_y = self.spline.calc_position(s)
        ref_yaw = self.spline.calc_yaw(s)

        # Apply lateral offset (d)
        x = ref_x + d * np.cos(ref_yaw + np.pi / 2)
        y = ref_y + d * np.sin(ref_yaw + np.pi / 2)

        return x, y

# Reference path
reference_x = df_cart['x'].values
reference_y = df_cart['y'].values

# Initialize the FrenetConverter with the reference path
converter = FrenetConverter(reference_x, reference_y)

# Convert a set of random Cartesian points to Frenet and back to Cartesian
# cartesian_points = [(reference_x[0], reference_y[0]), (reference_x[5], reference_y[5]), (reference_x[10], reference_y[10]),
#                     (reference_x[15], reference_y[15]), (reference_x[20], reference_y[20]),
#                     (reference_x[25], reference_y[25]), (reference_x[30], reference_y[30]),
#                     (reference_x[35], reference_y[35]), (reference_x[40], reference_y[40]),
#                     (reference_x[45], reference_y[45]), (reference_x[50], reference_y[50]),
#                     (reference_x[55], reference_y[55]), (reference_x[60], reference_y[60]),
#                     (reference_x[65], reference_y[65]), (reference_x[70], reference_y[70]),
#                     (reference_x[75], reference_y[75]), (reference_x[80], reference_y[80]),
#                     (reference_x[85], reference_y[85]), (reference_x[90], reference_y[90]),
#                     (reference_x[95], reference_y[95]), (reference_x[99], reference_y[99])]

# frenet_points = [converter.cartesian_to_frenet(x, y) for x, y in cartesian_points]

s = df_frenet['f_waypoints_s'].values
d = df_frenet['f_waypoints_d'].values
frenet_points = [(s[i], d[i]) for i in range(len(s))]

reconverted_points = [converter.frenet_to_cartesian(s, d) for s, d in frenet_points]

# Plotting
plt.figure(figsize=(8, 8))

# Plot the reference path
plt.plot(reference_x, reference_y, 'g--', label='Reference Path')

# Plot the original Cartesian points
# cartesian_points = np.array(cartesian_points)
# plt.scatter(cartesian_points[:, 0], cartesian_points[:, 1], color='blue', label='Original Cartesian Points')

# Plot the reconverted Cartesian points
reconverted_points = np.array(reconverted_points)
plt.scatter(reconverted_points[:, 0], reconverted_points[:, 1], color='red', marker='x', label='Reconverted Cartesian Points')

# Save reconverted Cartesian points to CSV

# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent
# print(f"Script directory: {script_dir}")

# Navigate up to the project root
project_root = script_dir.parent
# print(f"Project root: {project_root}")

# Build the path to the 'data' directory
data_dir = project_root  / 'data'
# print(f"Data directory: {data_dir}")
frenet_data = {'x_ref': reconverted_points[:, 0], 'y_ref': reconverted_points[:, 1]}
# extract frenet data to .txt file and .csv file at data folder
df = pd.DataFrame(frenet_data)

# Save the data to a .txt file
df.to_csv(data_dir / 'tracking_frenet_data.csv', index=False)
# Formatting the plot
plt.title('Frenet to Cartesian')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()