import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class EightShapedRoad:
    def __init__(self, a=12, b=6, road_half_width=0.5):
        """ 
        The laminscate of bernaulli is a figure-eight shaped curve defined by the equation:
        x = a * cos(t)
        y = b * cos(t)sin(t)
        where:
        t is the parameter that varies from 0 to 2*pi.
        a = width factor
        b = height factor
        road_half_width = half of the road width
        """
        self.a = a 
        self.b = b 
        self.road_half_width = road_half_width 

        t_vals = np.linspace(0, 2* np.pi, 200)  # Parameter values (t)

        # Using the parametric equations of the curve to compute x and y waypoints
        self.pos_x = np.round(self.a * np.sin(t_vals), 3)
        self.pos_y = np.round(self.b * np.sin(t_vals) * np.cos(t_vals), 3)

        # Compute derivatives of x and y to find the direction of the curve
        dx_dt = np.gradient(self.pos_x, t_vals)
        dy_dt = np.gradient(self.pos_y, t_vals)

        # Compute the differential arc length ds = sqrt(dx_dt^2 + dy_dt^2)
        ds = np.sqrt(dx_dt**2 + dy_dt**2)

        # Compute the cumulative arc length s as the arc length parameter
        self.s_vals = np.cumsum(ds)  # Cumulative sum of ds gives the arc length

        # Normalize the direction vectors to unit length
        dx_dt = dx_dt / ds
        dy_dt = dy_dt / ds

        # Compute the perpendicular vectors for offsetting the road boundaries
        self.pos_left_x = self.pos_x - self.road_half_width * dy_dt
        self.pos_left_y = self.pos_y + self.road_half_width * dx_dt

        self.pos_right_x = self.pos_x + self.road_half_width * dy_dt
        self.pos_right_y = self.pos_y - self.road_half_width * dx_dt

        # Store dx/ds and dy/ds for theta and curvature calculation
        # dx_ds,dy_ds represent the unit tangent vector at each, showing the direction of the curve w.r.t ds (arc length)
        self.dx_ds = dx_dt
        self.dy_ds = dy_dt

        # Compute dtheta/ds (curvature) as the derivative of orientation along the curve
        # Curvature (kappa) is the magnitude of the rate of change of the unit tangent vector w.r.t arc length. = d(theta)/ds
        self.dtheta_ds = np.gradient(np.arctan2(self.dy_ds, self.dx_ds), self.s_vals)

    def road_plots(self):
        """Method to plot the road with central lane and boundaries."""
        plt.figure(figsize=(8, 8))
        plt.plot(self.pos_x, self.pos_y, 'b--', label='Central Lane', linewidth=2)  # Central lane
        plt.plot(self.pos_left_x, self.pos_left_y, 'g-', label='Left Boundary', linewidth=1)  # Left boundary
        plt.plot(self.pos_right_x, self.pos_right_y, 'g-', label='Right Boundary', linewidth=1)  # Right boundary

        # Fill the area between the left and right boundaries to represent the road
        plt.fill_betweenx(self.pos_y, self.pos_left_x, self.pos_right_x, color='gray', alpha=0.5, label='Road Surface')
        # plot starting points and direction vectors
        plt.plot(self.pos_x[0], self.pos_y[0], 'ko', label='Start Point')
        plt.plot(self.pos_x[5], self.pos_y[5], 'ko', label='Start Point')
        plt.quiver(self.pos_x[::10], self.pos_y[::10], self.dx_ds[::10], self.dy_ds[::10], scale=10, color='r', label='Direction Vectors')


        # Format plot
        plt.title('Figure-Eight Road')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

    def get_point(self):
        """
        Method to get the x, y, theta, and curvature at all points along the road.
        x and y are the coordinates of the road centerline.
        theta is heading angle of the road at each point.
        kappa is the curvature of the road at each point.
        """
       
        x = self.pos_x
        y = self.pos_y
        theta = np.arctan2(self.dy_ds, self.dx_ds)
        kappa = self.dtheta_ds
        arc_length = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        # print(np.cumsum(arc_length))
        arc_length = np.append(arc_length, arc_length[-1])

        return x, y, theta, kappa, arc_length
    
    def export_data(self, filename='eight_shaped_road.csv'):
        # Method to export the road data to a CSV file
        x, y, theta, kappa, arc_length = self.get_point()
        data = np.vstack((x, y, theta, kappa, arc_length)).T
        np.savetxt(filename, data, delimiter=',', header='x,y,theta,kappa,arc_length', comments='')


# test
road = EightShapedRoad(a=40, b=80, road_half_width=0.5)
road.road_plots()

# Determine the directory of the current script
script_dir = Path(__file__).resolve().parent
print(f"Script directory: {script_dir}")

# Navigate up to the project root
project_root = script_dir.parent
print(f"Project root: {project_root}")

# Build the path to the 'data' directory
data_dir = project_root  / 'data'
print(f"Data directory: {data_dir}")

road.export_data(data_dir/'eight_shaped_road.csv')


