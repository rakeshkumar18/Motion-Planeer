"""
Clothoid also known as Euler Spiral, is a curve whoes curvature is a linear function of its arc length.
The key feature of the clothoid is that it provides as smooth transition between straight lines and a circular arc by gradually varying the curvature.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Clothoid:
    def __init__(self, x0, y0, psi0, k0, alpha):
        """
        x0, y0: initial position of the curve:  unit m
        psi0: initial heading of the curve: unit rad
        k0: initial curvature of the curve: unit 1/m
        alpha: rate of change of curvature of the curve: unit 1/m^2
        """
        self.x0 = x0
        self.y0 = y0
        self.psi0 = psi0
        self.k0 = k0
        self.alpha = alpha
        

    def clothoid_rhs(self, state, s):
        k = self.get_curvature(s)
        alpha = self.alpha
        psi = self.get_heading(s)
        x, y, theta = state[0], state[1], state[2]
        return np.array([np.cos(psi), np.sin(psi), k + alpha * s])
    
    def get_points(self, s):
        return odeint(self.clothoid_rhs, [self.x0, self.y0, psi0], s)

    def get_curvature(self, s):
        return self.k0 + self.alpha * s

    def get_heading(self, s):
        return self.psi0 + self.k0 * s + 0.5 * self.alpha * s**2

    def plot(self, s):
        states= self.get_points(s)
        x = states[:, 0]
        y = states[:, 1]
        plt.plot(x, y)
        plt.axis('equal')
        plt.show()

    def plot_road_with_clothoid_as_central_line(self, s):
        states= self.get_points(s)
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        # Wrap theta values to [-pi, pi]
        for i in range(len(theta)):
            theta[i] = (theta[i] + np.pi) % (2 * np.pi) - np.pi


        lane_width = 2.0 # Width
        x_left = x - 0.5 * lane_width * np.sin(theta)
        y_left = y + 0.5 * lane_width * np.cos(theta)

        x_right = x + 0.5 * lane_width * np.sin(theta)
        y_right = y - 0.5 * lane_width * np.cos(theta)

        # Plot the road
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='Central Lane')  # Central lane (clothoid)
        plt.plot(x_left, y_left, 'r--', label='Left Boundary') # Left boundary
        plt.plot(x_right, y_right, 'g--', label='Right Boundary') # Right boundary
        plt.fill_betweenx(y, x_left, x_right, color='gray', alpha=0.5) # Fill road area

        plt.title('Road Segment Using Clothoid as Central Lane')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    x0, y0 = 0, 0
    psi0 = 0.1
    k0 = 0.0
    alpha = -0.01
    clothoid = Clothoid(x0, y0, psi0, k0, alpha)
    s = np.linspace(0, 15, 300)
    clothoid.plot_road_with_clothoid_as_central_line(s)