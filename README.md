# Potential Fields Lab

Hands-on sessions on potential-field navigation with the TurtleBot3 Burger and ROS 2 Humble.
Smart Robots Design Lab, Tohoku University.

| Session | Topic | Guide |
|---|---|---|
| 1 | 1D potential fields: stopping in front of an obstacle | [Session 1](#session-1-1d-potential-fields) (below) |
| 2 | 2D potential fields: attraction, repulsion and traps | [docs/session2_2d_fields.md](docs/session2_2d_fields.md) |

**Already have the package from an earlier session?** Follow section 1 of the
newest session guide: pull the new code and do a clean build.

---

## Installation (laptop, first time only)

**1. Install TurtleBot3 packages**
```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-msgs
```

**2. Download and build this package**
```bash
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src
git clone https://github.com/smartrobotsdesignlab/potential_fields_lab.git
cd ~/turtlebot3_ws
colcon build --packages-select potential_fields_lab --symlink-install
source install/setup.bash
```

**3. Set environment variables** (add to `~/.bashrc`)
```bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30          # Change to 31 for Group 2
```

Then reload:
```bash
source ~/.bashrc
```

---

## Session 1: 1D potential fields

### Running the experiments

**On the Robot Pi** (SSH in first):
```bash
ssh on Robot 1 or 2
ros2 launch turtlebot3_bringup robot.launch.py
```

**On your Laptop**: run the experiment:
```bash
ros2 launch potential_fields_lab lab.launch.py config:=exp1_baseline
ros2 launch potential_fields_lab lab.launch.py config:=exp2_no_damping
ros2 launch potential_fields_lab lab.launch.py config:=exp3_weak_repulsion
ros2 launch potential_fields_lab lab.launch.py config:=exp4_strong_repulsion

```

Stop any experiment with `Ctrl+C`.

---

### Plotting results

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

## Repository structure

```
potential_fields_lab/
├── config/
│   ├── exp1_*.yaml ... exp5_*.yaml      Session 1 experiments
│   ├── scenario1_single.yaml ...        Session 2 scenarios 1 to 5
│   └── pf2d.rviz                        Session 2 RViz layout
├── docs/
│   └── session2_2d_fields.md            Session 2 guide
├── launch/
│   └── lab.launch.py                    Session 1 launcher
├── potential_fields_lab/
│   ├── potential_field_1d.py            Session 1 node
│   ├── pf_logger.py                     Session 1 logger
│   ├── potential_field_2d.py            Session 2 node
│   └── fake_diffdrive.py                Session 2 simulator (no robot needed)
├── scripts/
│   ├── plot_results.py                  Session 1 plots
│   ├── pf_lecture_plots.py              Session 2 theory plots
│   └── pf_scenario_builder.py           Session 2 interactive scenario builder
├── package.xml
└── setup.py
```

---

## Parameters

All parameters are set in the YAML config files. Check the config folder and files for details.
