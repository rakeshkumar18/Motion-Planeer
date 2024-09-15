# Structure of Repo of Frenet-Motion-Planner

├── Frenet_Seret_Frame │ ├── LICENSE │ ├── README.md │ ├── launch.sh │ ├── requirements.txt │ └── src │ ├── controller │ │ ├── Dynamics.py │ │ ├── config.py │ │ ├── enu_waypoints.txt │ │ ├── ilqr_maths_main.py │ │ ├── objective_func.py │ │ └── tracking_simulation.py │ ├── data │ │ ├── animation.gif │ │ ├── eight_shaped_road.csv │ │ ├── frenet_data.txt │ │ ├── frenet_frame_with_obstacle.csv │ │ ├── simulation.gif │ │ └── tracking_frenet_data.csv │ ├── main │ │ ├── frenetoptimaltrajectory.py │ │ └── tf_frennet_cartesian.py │ ├── temp │ │ └── tf.py │ └── utils │ ├── clothoid.py │ ├── cubic_spline_planner.py │ ├── eight_shaped_trajectory.py │ ├── frenet.ipynb │ └── quintic_quartic_polynomials_planner.py └── init.py

# Steps to Run:

git clone the code from branch "master" both without obstacle and "" with obstacle "tracking-with-obstacle"

## Setup Virtual Environment
Steps to create and activate a virtual environment:

# Navigate to the project directory:
    cd /path/to/your/project
# Create Virtual env python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

# Frenet Frame Motion Planner

This project implements a Frenet optimal trajectory planner and a tracking simulation. The project is organized into multiple Python scripts, each serving a specific function in the trajectory planning and tracking pipeline.

## How to Run the Project

The `launch.sh` script automates the execution of three main Python scripts sequentially. Before running the scripts, ensure you have set up the virtual environment and installed the required dependencies.

### 1. Setup

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd Frenet_Frame_Motion_Planner

2. **Creating Virtual Env**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install Dependcies**:

    ```bash
    pip install -r requirements.txt

### 2. Running the Project

1. **Create Executable for launch.sh**:

   ```bash
   chmod +x launch.sh
2. **Run launch.sh from dir Frenet_Seret_Frame**:

   ```bash
   ./launch.sh

## The script will execute the following Python scripts in sequence:


# frenetoptimaltrajectory.py: 
    This script generates the optimal trajectory in the Frenet frame based on the reference trajectory and obstacles in Cartesian coordinates.
![Alt text](Frenet_Seret_Frame/src/data/animation.gif)

# tf_frennet_cartesian.py:
    This script converts the optimal Frenet trajectory back into Cartesian coordinates for tracking.
![Alt text](path/to/animation.gif)

#   tracking_simulation.py: 
    This script simulates the vehicle tracking the generated trajectory using a control algorithm
![Alt text](/Users/rk/Documents/Github/Motion-Planner/Frenet_Seret_Frame/src/data/trajectories.png)
