# Potential Fields 1D Lab
ROS2 Humble — TurtleBot3 Burger

## Setup
```bash
cd ~/turtlebot3_ws/src
git clone <your_repo_url>
cd ~/turtlebot3_ws
colcon build --packages-select potential_fields_lab
source install/setup.bash
```

## Run on real robot
```bash
# On Pi
ros2 launch turtlebot3_bringup robot.launch.py

# On laptop
ros2 run potential_fields_lab potential_field_1d
```

## Parameters
Edit `potential_fields_lab/potential_field_1d.py`:
- `k_att` — attractive gain
- `k_rep` — repulsive gain  
- `influence_radius` — obstacle influence distance (m)
- `goal_distance` — target stop distance (m)
- `k_damp` — damping coefficient
