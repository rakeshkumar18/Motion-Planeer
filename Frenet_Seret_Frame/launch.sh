#!/bin/bash

# Running the first Python file
echo "Running frenetoptimaltrajectory.py"
python3 src/main/frenetoptimaltrajectory.py

# Check if the first script executed successfully
if [ $? -eq 0 ]; then
    echo "frenetoptimaltrajectory.py completed successfully."
else
    echo "Error running first_script.py"
    exit 1
fi

# Running the second Python file
echo "tf_frennet_cartesian.py"
python3 src/main/tf_frennet_cartesian.py

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "tf_frennet_cartesian.py completed successfully."
else
    echo "Error running second_script.py"
    exit 1
fi

# Running the third Python file

echo "tracking_simulation.py"

python3 src/controller/tracking_simulation.py

# Check if the third script executed successfully
if [ $? -eq 0 ]; then
    echo "tracking_simulation.py completed successfully."
else
    echo "Error running third_script.py"
    exit 1
fi

