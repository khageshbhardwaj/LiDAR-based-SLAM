# LiDAR-based-SLAM
The project involves creating a comprehensive robot localization and mapping system by using IMU, encoder and LiDAR data. The 
instructor provides the data for the project. However, the code could be used for any general data provided by
the user.

# Part 1: Odometry-based Trajectory
This part integrates encoder and IMU measurements for initial odometry estimation. Using the Euler discretization,
the program finds the robot's odometry trajectory. It also plots the odometry trajectory to visualize it.

# Part 2: ICP Optimized Trajectory
Oometry trajectory is further refined through LiDAR scan matching utilizing the Iterative Closest Point (ICP) algorithm. 
This code snippet utilises odometry trajectory for pose initialization and finds the optimized trajectory using the ICP
algorithm. It also plots the ICP-optimized trajectory.

# Part 3: 2D Occupancy Grid and Texture Mapping
Additionally, algorithms are developed to generate 2-D occupancy maps and texture maps of the environment by processing 
LiDAR scans and RGBD images from a Kinect sensor. This uses the ICP-optimized pose to generate the maps.

# Part 4: GTSAM Optimization Trajectory
To enhance trajectory estimates, the system utilizes loop-closure constraints, optimizing the robot's path using the 
GTSAM (Georgia Tech Smoothing and Mapping) library. This integrated approach aims to provide accurate and robust 
localization and mapping capabilities for autonomous robotic navigation in diverse environments.
To construct a loop closure/factor (at a regular interval), the program uses the ICP algorithm in between two nodes. 
This generates further optimized trajectories using the GTSAM library.

# How to run the code?
Run the [project2_part1_3.py] code. All the respective plots will be saved in the main folder. Provide a suitable path 
to access the data. Parts 1 to 3 are given in this single code file named [project2_part1_3.py]. The code is arranged 
sequentially from Part 1 to Part 3.

To generate the GTSAM-optimized trajectory, grid, and texture maps, use the [project2_part4_GTSAM.py] [factor_icp.py] codes.
This code has to be run in wsl or ubuntu. It is arranged as follows: The first section generates a Factor Graph, the second
generates the occupancy grid with a comparison of the trajectories, and the last section is a texture map.
This file calls upon another file named [factor_icp.py]; therefore, keep both files in the same folder.

# Dependencies
Mentioned under the first part of the code under the [import] section.

