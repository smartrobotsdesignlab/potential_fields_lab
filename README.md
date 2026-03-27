

---


## Installation (Laptop only)

**1. Install TurtleBot3 packages**
```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-msgs
```

**2. Clone and build this package**
```bash
cd ~/turtlebot3_ws/src
git clone git@github.com:smartrobotsdesignlab/potential_fields_lab.git
cd ~/turtlebot3_ws
colcon build --packages-select potential_fields_lab --symlink-install
source install/setup.bash
```

**3. Set environment variables** — add to `~/.bashrc`
```bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30          # Change to 31 for Group 2
```

Then reload:
```bash
source ~/.bashrc
```

---

## Running Experiments

**On the Robot Pi** (SSH in first):
```bash
ssh on Robot 1 or 2
ros2 launch turtlebot3_bringup robot.launch.py
```

**On your Laptop** — run the experiment:
```bash
ros2 launch potential_fields_lab lab.launch.py config:=exp1_baseline
ros2 launch potential_fields_lab lab.launch.py config:=exp2_no_damping
ros2 launch potential_fields_lab lab.launch.py config:=exp3_weak_repulsion
ros2 launch potential_fields_lab lab.launch.py config:=exp3_strong_repulsion




```



Stop any experiment with `Ctrl+C`.

---

## Plotting Results

Logs are saved automatically to `~/pf_logs/` after each experiment.

**Single experiment:**
```bash
python3 src/potential_fields_lab/scripts/plot_results.py --exp exp1_baseline
```

**Compare all four:**
```bash
python3 src/potential_fields_lab/scripts/plot_results.py \
  --compare exp1_baseline exp2_no_damping exp3_weak_repulsion exp4_strong_repulsion
```

---

## Repository Structure

```
potential_fields_lab/
├── config/
│   ├── exp1_baseline.yaml
│   ├── exp2_no_damping.yaml
│   ├── exp3_weak_repulsion.yaml
│   └── exp4_strong_repulsion.yaml
├── launch/
│   └── lab.launch.py
├── potential_fields_lab/
│   ├── potential_field_1d.py     
│   └── pf_logger.py              
├── scripts/
│   ├── plot_results.py           
│   └── plot_results_dark.py      
├── package.xml
└── setup.py
```

---

## Parameters

All parameters are set in the YAML config files. check the config folder and files for details. 
