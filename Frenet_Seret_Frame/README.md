# Structure of Repo of Frenet-Motion-Planner
├── Frenet_Seret_Frame
│   ├── LICENSE
│   ├── README.md
│   ├── launch.sh
│   ├── requirements.txt
│   └── src
│       ├── controller
│       │   ├── Dynamics.py
│       │   ├── config.py
│       │   ├── enu_waypoints.txt
│       │   ├── ilqr_maths_main.py
│       │   ├── objective_func.py
│       │   └── tracking_simulation.py
│       ├── data
│       │   ├── animation.gif
│       │   ├── eight_shaped_road.csv
│       │   ├── frenet_data.txt
│       │   ├── frenet_frame_with_obstacle.csv
│       │   ├── simulation.gif
│       │   └── tracking_frenet_data.csv
│       ├── main
│       │   ├── frenetoptimaltrajectory.py
│       │   └── tf_frennet_cartesian.py
│       ├── temp
│       │   └── tf.py
│       └── utils
│           ├── clothoid.py
│           ├── cubic_spline_planner.py
│           ├── eight_shaped_trajectory.py
│           ├── frenet.ipynb
│           └── quintic_quartic_polynomials_planner.py
└── __init__.py

# Steps to Run:
git clone the code from branch "tracking-with-obstacle"

## Setup Virtual Environment

Before running the project, it's recommended to use a virtual environment to manage dependencies.

### Steps to create and activate a virtual environment:

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/your/project
2. Create Virtual env 
    python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt


# Run 
./launch.sh

## launch.sh has three script:
1. python3 src/main/frenetoptimaltrajectory.py 
    It takes reference trajectory in cartesian coordinate and obstacles, provide the best trjectory.
![Alt text](Frenet_Seret_Frame/src/data/animation.gif)

2. python3 src/main/tf_frennet_cartesian.py
    It converts the frenet optimal trjectory into cartesian coordinates to track by a controller
![Alt text](Frenet_Seret_Frame/src/data/trajectories.png)
3. python3 src/controller/tracking_simulation.py
    It simulates the vehicle to track the generated trajectory
![Alt text](Frenet_Seret_Frame/src/data/simulation.gif)

