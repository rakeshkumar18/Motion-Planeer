import numpy as np

class quintic_polynomial:

    # __init__ is a reseved method in python classes. It is known as a constructor in object oriented concepts. 
    # This method called when an object is created from the class and it allow the class to initialize the attributes of a class.
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        """
        Args:
        
        xs : initial position
        vxs : initial velocity
        axs : initial acceleration
        xe : end position
        vxe : end velocity
        axe : end acceleration
        T : tme to go to end position

        """

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):                             # position w.r.t time
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):                  # velocity w.r.t time
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):                   # acceleration w.r.t time      
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):                       # jerk w.r.t time
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt

class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        """
        Args:
            xs: Initial position (m)
            vxs: Initial velocity (m/s)
            axs: Initial acceleration (m/s^2)
            vxe: Final velocity (m/s)
            axe: Final acceleration (m/s^2)
            time: Time to reach the final state (s)
        """

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):                           # position w.r.t time
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):          # velocity w.r.t time
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):      # acceleration w.r.t time
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):        # jerk w.r.t time
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt