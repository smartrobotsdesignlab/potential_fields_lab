# Session 2: 2D Potential Fields on the TurtleBot3

Step-by-step guide for the hands-on session. Work through the sections in order.
All commands are run on your **laptop** unless a step says **robot (Pi)**.

1. Update from Session 1
2. Simulation: scenarios 1 to 5
3. Real robot runs
4. Plotting your runs
5. Playing with the parameters
6. Troubleshooting

---

## 1. Update from Session 1

Your laptop already has this package from Session 1. Session 2 adds new nodes,
config files and scripts, so you need to pull the new code and **clean build**.
A normal build on top of the old one can fail or run old code.

### 1.1 Pull the new code

```bash
cd ~/turtlebot3_ws/src/potential_fields_lab
git status
```

If `git status` lists files you changed in Session 1 (for example a config you edited),
put them aside first, then pull:

```bash
git stash          # only if git status showed changed files
git pull
```

If `git pull` asks for a password or says `Permission denied (publickey)`,
switch the repository to HTTPS once and pull again:

```bash
git remote set-url origin https://github.com/smartrobotsdesignlab/potential_fields_lab.git
git pull
```

**Do not have the package yet?** Clone it instead:

```bash
cd ~/turtlebot3_ws/src
git clone https://github.com/smartrobotsdesignlab/potential_fields_lab.git
```

### 1.2 Clean build

Delete the old build output of this package only, then build again:

```bash
cd ~/turtlebot3_ws
rm -rf build/potential_fields_lab install/potential_fields_lab
colcon build --symlink-install --packages-select potential_fields_lab
source install/setup.bash
```

`rm -rf` here only removes files that the build creates. Your source code in
`src/` is not touched.

### 1.3 Check that the new programs are there

```bash
ros2 pkg executables potential_fields_lab
```

You should see these lines (order may differ):

```
potential_fields_lab fake_diffdrive
potential_fields_lab pf_logger
potential_fields_lab potential_field_1d
potential_fields_lab potential_field_2d
```

If `potential_field_2d` or `fake_diffdrive` is missing, the build did not pick up
the new code. Repeat 1.2 and make sure you ran `source install/setup.bash`.

### 1.4 Every new terminal

Each new terminal must load the workspace before running anything:

```bash
source ~/turtlebot3_ws/install/setup.bash
```

Check that your group's settings from Session 1 are still in `~/.bashrc`:

```bash
echo $TURTLEBOT3_MODEL $ROS_DOMAIN_ID     # expect: burger 30  (or 31 for Group 2)
```

If nothing is printed, add them again as in the main README.

---

## 2. Simulation: scenarios 1 to 5

The simulator (`fake_diffdrive`) is a stand-in for the robot. Everything else is
exactly what runs on the real TurtleBot. Do the scenarios in order.

For each scenario: **predict first, then run, then answer the questions.**
Write your answers down; we will discuss them together.

### 2.1 Start the simulator and RViz (once)

Open two terminals and leave them running for all five scenarios.

**Terminal 1: simulator**
```bash
ros2 run potential_fields_lab fake_diffdrive
```

**Terminal 2: RViz**
```bash
rviz2 -d ~/turtlebot3_ws/src/potential_fields_lab/config/pf2d.rviz
```

RViz shows the goal (green), the obstacles (red), each obstacle's influence
circle d0 (orange), the force arrow on the robot and the robot's path (teal).

> Run only **one** simulator. Two copies fight each other and the robot jumps around.

### 2.2 How to run a scenario

**Terminal 3: the potential field node.** Change only the file name:
```bash
ros2 run potential_fields_lab potential_field_2d --ros-args \
  --params-file ~/turtlebot3_ws/src/potential_fields_lab/config/scenario1_single.yaml
```

**Terminal 4: the theory plot.** Change only the number:
```bash
python3 ~/turtlebot3_ws/src/potential_fields_lab/scripts/pf_lecture_plots.py --scenario 1
```

When a run ends, press `Ctrl+C` in terminal 3 and start the next scenario there.
The simulator keeps running; the node treats the robot's current position as the
new start (0, 0).

Terminal 3 prints a line like this every 0.1 s:
```
t=  2.5s | pos=( 0.36, 0.00) | |F|= 0.97 | dg=1.64 | v=0.15 w=+0.00
```

| Value | Meaning |
|---|---|
| `pos` | robot position (m) |
| `\|F\|` | strength of the total force |
| `dg` | distance to the goal (m) |
| `v`, `w` | forward speed (m/s) and turn rate (rad/s) sent to the wheels |

### 2.3 The scenarios

Open the config file of each scenario and look at the goal and the obstacles
**before** you run it.

**Scenario 1: one obstacle** (`scenario1_single.yaml`, `--scenario 1`)
- Predict: which side will the robot pass the obstacle on?
- Run it. Where does it end, and how long did it take?
- Look at `|F|` at the start and near the end. Why is it 1.00 at the start?
  Why does it become equal to `dg` near the end?

**Scenario 2: gap** (`scenario2_gap.yaml`, `--scenario 2`)
- The gap between the two obstacles is 0.6 m and the robot is 0.18 m wide.
  Predict: will it go through?
- Run it. What happens? Where exactly does it stop?
- At that point, which forces are acting on the robot? Draw them.

**Scenario 3: canyon** (`scenario3_canyon.yaml`, `--scenario 3`)
- Where is the goal compared with the wall? Predict what the robot will do.
- Run it. Describe what the robot does. (Stop it with `Ctrl+C` if it does not stop.)
- How is this different from scenario 2?

**Scenario 4: obstacle on the line** (`scenario4_collinear.yaml`, `--scenario 4`)
- The obstacle sits exactly on the line to the goal. Predict: left, right, or neither?
- Run it. What happens?
- Imagine pushing the robot 1 cm to the side at the point where it stopped.
  What would happen next? Would the same push help in scenario 2?

**Scenario 5: escape** (`scenario5_escape.yaml`, `--scenario 5`)
- Compare the file with scenario 4. What is the only difference?
- Run it. What changed?
- Why can the robot no longer get stuck at the same point?

### 2.4 Record your results

| Scenario | Your prediction | What happened | End position (m) | Time (s) |
|---|---|---|---|---|
| 1 Single | | | | |
| 2 Gap | | | | |
| 3 Canyon | | | | |
| 4 On the line | | | | |
| 5 Escape | | | | |

Keep this table. You will fill the same one on the real robot in section 3.

---

## 3. Real robot runs

Now the same five scenarios on the TurtleBot3. The robot does **not** see the
obstacles in this session: their positions come from the config file. The objects
you place on the floor only show where the virtual obstacles are.

### 3.1 Before you start

1. **Stop the simulator** (`Ctrl+C` in terminal 1). The simulator and the real
   robot must never run at the same time.
2. **Floor setup.** You need about 3 m x 1.5 m of free floor.
   - Mark the start point with tape and draw a short arrow for the forward (+x) direction.
   - Measure from the start mark and place a light marker (paper cup, small
     cylinder) at each obstacle position in the config file. Use soft or light
     objects: if the robot touches one, nothing should break.
   - Mark the goal with tape.
3. **Your ROS_DOMAIN_ID must be the same as your robot's.** Check with
   `echo $ROS_DOMAIN_ID`.

### 3.2 Start the robot

**Robot (Pi):** log in to your group's robot and start it:
```bash
ssh <user>@<robot address>          # your instructor gives you the address
ros2 launch turtlebot3_bringup robot.launch.py
```

**Laptop:** check that the robot is visible:
```bash
ros2 topic list | grep odom          # must print /odom
```

Then start RViz as in section 2 (terminal 2).

### 3.3 Run a scenario

1. Put the robot on the start mark, facing the arrow. **The robot's position and
   direction at the moment you start the node become (0, 0) and +x.**
2. Start the node exactly as in section 2 (terminal 3), with the scenario's file.
3. Keep a hand near the robot. When the run ends, press `Ctrl+C` in terminal 3.
4. Measure where the robot really stopped with a tape measure, from the start mark.
5. Carry the robot back to the start mark and start the next scenario.

**Emergency stop.** If the robot keeps moving after `Ctrl+C`, pick it up and run:
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

### 3.4 Questions

Before each run, look at your simulation result from section 2.4.

- **Scenario 1:** Predict: will the real robot end at the same point as the simulation?
  After the run: compare the `GOAL REACHED` position in the log with your tape
  measurement. Are they the same? If not, which one is wrong, and why?
- **Scenario 1 again:** run it a second time without changing anything. Same end point?
- **Scenario 2:** does the real robot also get stuck? Where, compared with the simulation?
- **Scenario 3:** compare with the simulation. Anything different?
- **Scenario 4:** watch carefully. Does the real robot behave like the simulation?
  If not, what is different on the real robot that the simulation does not have?
- **Scenario 5:** does the robot pass the obstacle on the same side as in the simulation?
- **Overall:** the robot never saw your markers. What would happen if you moved a
  marker by 30 cm without changing the config file? What does this tell you about
  this way of avoiding obstacles?

### 3.5 Record your results

| Scenario | Simulation end (m) | Real: log end (m) | Real: tape end (m) | Same behaviour as sim? |
|---|---|---|---|---|
| 1 Single | | | | |
| 1 Single (2nd run) | | | | |
| 2 Gap | | | | |
| 3 Canyon | | | | |
| 4 On the line | | | | |
| 5 Escape | | | | |

---

## 4. Plotting your runs

The theory plot shows the force field (arrows), the potential (contour lines),
each obstacle's d0 circle and the path an ideal robot would take.

```bash
cd ~/turtlebot3_ws/src/potential_fields_lab/scripts
python3 pf_lecture_plots.py --scenario 2                  # one scenario
python3 pf_lecture_plots.py --all                         # all five side by side
python3 pf_lecture_plots.py --scenario 2 --field-only     # field only, no path
python3 pf_lecture_plots.py --all --save ~/pf_figs        # save PNG + PDF
```

After each real run, take a screenshot of RViz with the path visible and put it
next to the theory plot.

- Does the real path have the same shape as the theory path?
- Where do they start to differ? What do you think causes it?
- Look at the arrows near the point where the robot stopped in scenario 2.
  What do you notice about them?

---

## 5. Playing with the parameters

Now change the parameters and see what happens. **Do not edit the original config
files.** Work on copies in your own folder:

```bash
mkdir -p ~/pf_configs
cp ~/turtlebot3_ws/src/potential_fields_lab/config/scenario2_gap.yaml ~/pf_configs/my_gap.yaml
```

Open `~/pf_configs/my_gap.yaml` in a text editor, change one value, save, and run:

```bash
ros2 run potential_fields_lab potential_field_2d --ros-args --params-file ~/pf_configs/my_gap.yaml
```

> Write numbers with a decimal point: `1.0`, not `1`. Otherwise the node refuses the file.

**Scenario builder.** To see the expected result before you run, use the builder.
Click to add obstacles, shift+click to move the goal, and use the sliders:

```bash
python3 ~/turtlebot3_ws/src/potential_fields_lab/scripts/pf_scenario_builder.py --out ~/pf_configs
```

`Save YAML` writes a config into `~/pf_configs` that you can run directly.

Change **one parameter at a time** and predict before each run.

### 5.1 Influence radius d0 (`influence_radius`)

Try 0.4, 0.7 (default) and 1.0 on scenario 1 and scenario 2.

- Scenario 1: does the robot start to turn earlier or later? Does it pass closer or further?
- Scenario 2: can you find a d0 that lets the robot through the gap?

### 5.2 Repulsion strength k_rep (`k_rep`)

Try 0.1, 0.5 (default) and 2.0 on scenarios 1 and 2.

- How close does the robot get to the obstacle?
- Does the robot still get stuck in the gap? At the same place?

### 5.3 Attraction strength k_att (`k_att`)

Try 0.5, 1.0 (default) and 2.0 on scenarios 1 and 2.

- Does the robot drive faster with a larger k_att? (Look at `v` in the log.)
- Where does it stop in the gap now?

### 5.4 Rotational field (`rotational_field`, `k_rot`)

- Switch `rotational_field: true` in your copy of scenario 2. Does it escape the gap?
- Try it in scenario 3 (canyon). Does it help there?

### 5.5 Record your results

| Scenario | Parameter | Value | Prediction | What happened | End (m) |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Final question: from everything you tried, can you find **one** set of parameters
that solves all five scenarios? Why or why not?

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Package 'potential_fields_lab' not found` or `No executable found` | `source ~/turtlebot3_ws/install/setup.bash`. If still missing, repeat the clean build (1.2). |
| Build fails after `git pull` (`can't copy ... doesn't exist`) | Clean build (1.2): delete `build/potential_fields_lab` and `install/potential_fields_lab` first. |
| `UnknownROSArgsError` | Check the spelling: `--params-file` (with an **s**). |
| Node refuses the YAML file (parameter type error) | Write numbers with a decimal point: `1.0`, not `1`. |
| Robot says `GOAL REACHED` immediately | Restart the node. It must be started with the robot at the start mark. |
| Robot jumps around in RViz | Two simulators are running, or the simulator is running with the real robot. Stop all and start one. |
| `ros2 topic list` does not show `/odom` from the robot | Same `ROS_DOMAIN_ID` on laptop and robot? Bringup still running on the robot? Same Wi-Fi network? |
| Cannot connect to the robot with `ssh` | Robot switched on and booted (about 1 min)? Ask the instructor for the address. |
| Robot keeps moving after `Ctrl+C` | Pick it up, then `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"` |
| `git pull` asks for a password | See 1.1: switch to the HTTPS address. |
| `git pull` says your local changes would be overwritten | `git stash`, then `git pull`. |
