# LiDAR-based-SLAM


# Part1-Part3 [project2_part1_3.py]
Run the code from the main folder, i.e outside code folder. All the respective plots will get saved in main folder.
Keep the code inside the code folder or provide suitable path to access the data. From part 1 to part 3 are given in single code file named as [project2_part1_3.py]. Please provide the [Datadet] number, for which we are seeking the results. The code is arranged sequencially from Part 1 to Part 3. As .py file didnt allow to show the file, we have saved all the plots for trajectory, ICP otimised trajectory, occupancy grid map and texture frif map.

[Dependencies]: Mentioned under first part of the code under [import] section.

# Part 4 - GTSAM Optimization Trajectory [project2_part4_GTSAM.py] [factor_icp.py]
Keep this file inside the code folder or provide suitable path to access the data. The plots will get saved to the same folder. This code has to be run in wsl or ubuntu. Provide the [Datadet] number, for which we are seeking the results. Code is arranged as follows: First section generates Factor Graph, then the second generates the occupancy grid with comparison in the trajectories, and the last one section is texture map. The respective plots will get saved to the respective code location.

This file calls upon another file named [factor_icp.py], therefore keep both the files in the same folder.

[Dependencies]: Mentioned under first part of the code under [import] section.

