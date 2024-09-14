import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class FrenetFrame:
    def __init__(self, x_curve, y_curve, theta, s_curve, kappa):
        """
        Initialize the FrenetFrame class with reference curve data.
        
        Parameters:
        x_curve (np.array): x-coordinates of the reference curve.
        y_curve (np.array): y-coordinates of the reference curve.
        theta (np.array): Heading angles (tangents) of the reference curve.
        s_curve (np.array): Arc length along the reference curve.
        kappa (np.array): Curvature values of the reference curve.
        """
        self.x_curve = x_curve
        self.y_curve = y_curve
        self.theta = theta
        self.s_curve = s_curve
        self.kappa = kappa
        # Compute the tangent and normal vectors
        self.T = np.array([np.cos(self.theta), np.sin(self.theta)]).T
        self.N = np.array([-np.sin(self.theta), np.cos(self.theta)]).T

    def cartesian_to_frenet(self, x_points, y_points):
        """Convert Cartesian coordinates to Frenet coordinates (s, d)."""
        s_frenet = np.zeros_like(x_points)
        d_frenet = np.zeros_like(y_points)
        
        for i in range(len(x_points) - 1):
            # Find the closest point on the reference curve
            distances = np.sqrt((x_points[i] - self.x_curve)**2 + (y_points[i] - self.y_curve)**2)
            closest_idx = np.argmin(distances)
            # print(f"Closest index: {closest_idx}")
            # Frenet s is the arc length at the closest point
            s_frenet[i+1] = s_frenet[i] + self.s_curve[closest_idx]
            # print(f"s_frenet: {s_frenet[i]}")
            
            # Compute the lateral distance d (dot product with normal vector)
            delta = np.array([x_points[i] - self.x_curve[closest_idx], y_points[i] - self.y_curve[closest_idx]])
            d_frenet[i] = np.dot(delta, self.N[closest_idx])
        
        return s_frenet, d_frenet
    
    def frenet_to_cartesian(self, s, d):
        """
        Convert Frenet coordinates to Cartesian coordinates.
        
        Parameters:
        s (float): Longitudinal displacement along the reference curve (arc length).
        d (float): Lateral displacement perpendicular to the reference curve.
        s_dot (float): Longitudinal velocity.
        d_dot (float): Lateral velocity.
        s_ddot (float): Longitudinal acceleration.
        d_ddot (float): Lateral acceleration.
        
        Returns:
        dict: A dictionary with Cartesian position, velocity, and acceleration.
        """
        # Find the closest point on the reference curve (by arc length)
        closest_idx = np.argmin(np.abs(s - self.s_curve))

        # Position in Cartesian coordinates
        x_ref = self.x_curve[closest_idx]
        y_ref = self.y_curve[closest_idx]

        x = x_ref + d * self.N[closest_idx, 0]
        y = y_ref + d * self.N[closest_idx, 1]

        # # Velocity in Cartesian coordinates
        # x_dot = s_dot * self.T[closest_idx, 0] + d_dot * self.N[closest_idx, 0]
        # y_dot = s_dot * self.T[closest_idx, 1] + d_dot * self.N[closest_idx, 1]

        # # Acceleration in Cartesian coordinates
        # kappa_val = self.kappa[closest_idx]
        # x_ddot = (s_ddot - kappa_val * s_dot**2) * self.T[closest_idx, 0] + \
        #          (d_ddot + 2 * kappa_val * s_dot * d_dot + kappa_val * s_dot**2) * self.N[closest_idx, 0]
        # y_ddot = (s_ddot - kappa_val * s_dot**2) * self.T[closest_idx, 1] + \
        #          (d_ddot + 2 * kappa_val * s_dot * d_dot + kappa_val * s_dot**2) * self.N[closest_idx, 1]

        return {
            'x': x, 'y': y
        }


    def plot_cartesian_to_frenet(self, s_frenet, d_frenet, x_points, y_points):
        """Plot Cartesian and Frenet coordinates."""
        plt.figure(figsize=(12, 6))

        # Plot 1: Cartesian Coordinates (x, y)
        plt.subplot(1, 2, 1)
        plt.plot(self.x_curve, self.y_curve, label="Reference Curve (x, y)", color='blue')
        plt.scatter(x_points, y_points, c='red', label="Data Points")
        plt.title("Cartesian Coordinates (x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

        # Plot 2: Frenet Coordinates (s, d)
        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(self.s_curve), np.zeros_like(self.s_curve), label="Reference Curve (s)", color='blue')
        plt.scatter(s_frenet, d_frenet, c='red', label="Data Points in Frenet Frame")
        plt.title("Frenet Coordinates (s, d)")
        plt.xlabel("Arc Length (s)")
        plt.ylabel("Lateral Distance (d)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_frenet_to_cartesian(self, s, d, s_dot, d_dot, s_ddot, d_ddot):
        """Plot Cartesian coordinates from Frenet coordinates."""
        # Convert Frenet to Cartesian coordinates
        cartesian_coords = self.frenet_to_cartesian(s, d, s_dot, d_dot, s_ddot, d_ddot)
        
        # Plot Cartesian coordinates
        plt.figure(figsize=(6, 6))
        plt.plot(self.x_curve, self.y_curve, label="Reference Curve", color='blue')
        plt.scatter(cartesian_coords['x'], cartesian_coords['y'], c='red', label="Converted Point")
        plt.title("Cartesian Coordinates from Frenet Frame")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage:
# Load CSV data externally, not within the class
df = pd.read_csv('utils/eight_shaped_road.csv')

# Extract columns (x, y, theta, kappa, arc length)
x = df['x'].values
y = df['y'].values
theta = df['theta'].values
s = df['arc_length'].values
kappa = df['kappa'].values

# Create FrenetFrame instance
frenet_frame = FrenetFrame(x, y, theta, s, kappa)

# Convert Cartesian to Frenet coordinates
s_frenet, d_frenet = frenet_frame.cartesian_to_frenet(x, y)
# Plot the results
frenet_frame.plot_cartesian_to_frenet(s_frenet, d_frenet, x, y)

# Frenet coordinates (example values for demonstration)
s = s_frenet
d = d_frenet


# Convert Frenet to Cartesian coordinates
x_c = []
y_c = []
for i in range(len(s)):
    cartesian_coords = frenet_frame.frenet_to_cartesian(s[i], d[i])
    # print(f"Point {i+1} in Cartesian Coordinates:")
    # print(cartesian_coords)
    #plot the results
    # extract the x and y values
    x_c.append(cartesian_coords['x'])
    y_c.append(cartesian_coords['y'])
plt.figure(figsize=(8, 8))
plt.plot(x_c, y_c, 'b--', label='Central Lane', linewidth=2)  # Central lane
plt.show



