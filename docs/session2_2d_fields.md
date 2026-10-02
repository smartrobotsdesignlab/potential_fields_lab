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
