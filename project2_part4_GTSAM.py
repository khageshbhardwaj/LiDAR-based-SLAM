##########################################################################################################################
# Part 4
##########################################################################################################################

import gtsam
import numpy as np
from factor_icp import *
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
import cv2
import pathlib

current_path = pathlib.Path().resolve()
dataset = 21

with np.load(f"../data/Encoders{dataset}.npz") as data:
  encoder_counts = data["counts"] # 4 x n encoder counts
  encoder_stamps = data["time_stamps"] # encoder time stamps 

with np.load(f"../data/Hokuyo{dataset}.npz") as data:
  lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
  lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
  lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
  lidar_range_min = data["range_min"] # minimum range value [m]
  lidar_range_max = data["range_max"] # maximum range value [m]
  lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
  lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
  
with np.load(f"../data/Imu{dataset}.npz") as data:
  imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
  imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
  imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load(f"../data/Kinect{dataset}.npz") as data:
  disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
  rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

##################################################################################################################################

def create_factor_graph(poses, relative_poses):
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # Add prior factor for the first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    graph.add(gtsam.PriorFactorPose3(gtsam.symbol('x', 0), gtsam.Pose3(), prior_noise))
    initial_values.insert(gtsam.symbol('x', 0), gtsam.Pose3(poses[:, :, 0]))

    # Add odometry factors based on relative poses
    for i in range(1, relative_poses.shape[2]):
        print('GTSAM Itr No', i)
        odometry_factor = gtsam.BetweenFactorPose3(
            gtsam.symbol('x', i-1), gtsam.symbol('x', i),
            gtsam.Pose3(relative_poses[:, :, i-1]),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        )
        graph.add(odometry_factor)
        initial_values.insert(gtsam.symbol('x', i), gtsam.Pose3(poses[:, :, i]))

        # if the current pose is a multiple of 5 for loop closure
        if i % 5 == 0:
            # loop_closure_relative_pose is the result from ICP
            loop_closure_relative_pose = factorgraph_icp(i-4,i)

            loop_closure_factor = gtsam.BetweenFactorPose3(
                gtsam.symbol('x', i), gtsam.symbol('x', i-4),
                gtsam.Pose3(loop_closure_relative_pose),
                gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
            )
            graph.add(loop_closure_factor)

    return graph, initial_values

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))

def rotation_matrix_xyz(roll, pitch, yaw):
    # Calculate cosine and sine values
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    # Rotation matrix around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    # Rotation matrix around Y axis
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    # Rotation matrix around Z axis
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    # Combine the rotation matrices
    R = Rz @ Ry @ Rx

    return R


#################################################################################################################################

# From odometry for reference 
trajectory = np.load('trajectory.npy')

# Load pose and relative pose
T = np.load('pose_matrix_T.npy')
T_relative = np.load('pose_matrix_T_relative.npy')
lidar_pc = np.load('lidar_pc.npy')

# Create the factor graph and initial values
factor_graph, initial_values = create_factor_graph(T, T_relative)

# Optimize the factor graph using GTSAM
optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values)
result = optimizer.optimize()

# Result contains the optimized poses from GTSAM
num_poses = T.shape[2]-1  # T is the ICP poses array

# Initialize an array to store 4x4 transformation matrices
T_GTSAM = np.zeros((4, 4, num_poses))

for i in range(num_poses):
    # Extract the optimized Pose3 value for the current pose
    optimized_pose = result.atPose3(gtsam.symbol('x', i))

    # Get the rotation matrix and translation vector from the optimized pose
    rotation_matrix = optimized_pose.rotation().matrix()
    translation_vector = optimized_pose.translation()

    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector

    # Save the transformation matrix in the array
    T_GTSAM[:, :, i] = transform_matrix

# Save the array to a file
np.save('optimized_transformations.npy', T_GTSAM)

# Plot x vs y
fig3 = plt.figure()
plt.plot(trajectory[0, :], trajectory[1, :], label='Odometry Trajectory')
plt.plot(T[0, 3, :], T[1, 3, :], label='ICP Optimised Trajectory')
plt.plot(T_GTSAM[0, 3, :], T_GTSAM[1, 3, :], label='GTSAM Optimized Trajectory')
plt.legend()
plt.title('Comparison of Trajectories')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.grid(True)
plt.savefig('GTSAM Optimised Trajectory')

#########################################################################################################################

# Initialize the occupancy grid map
map_resolution = 0.05  # meters per cell
map_size = 1000  # size of the map in cells
xmin = -25
xmax = 25
ymin = -25
ymax = 25
occupancy_map = np.zeros((map_size, map_size), dtype=np.float32)

# point cloud coorfinates
x_lidar = lidar_pc[:,0]
y_lidar = lidar_pc[:,1]
z_lidar = lidar_pc[:,2]

for i in range(num_poses-1):
    print('Occupany Grid Itr No:', i)
    clm_ones = np.ones((x_lidar.shape[0]))    

    # convert position in the map frame here 
    Yt = np.stack((x_lidar[:,i],y_lidar[:,i],z_lidar[:,i],clm_ones), axis =1)

    # transformation to the world frame (rotated) with the GTSAM optimised poses
    Yr = Yt @ T_GTSAM[:,:,i].T

    xs0 = Yr[:,0]
    ys0 = Yr[:,1]

    x_start = int((T_GTSAM[:,:,i][0,3] - xmin) / map_resolution)
    y_start = int((T_GTSAM[:,:,i][1,3] - ymin) / map_resolution)

    for j in range(xs0.shape[0]):
        x_end = int((xs0[j] - xmin) / map_resolution)
        y_end = int((ys0[j] - ymin) / map_resolution)

        bresenham_path = bresenham2D(x_start, y_start, x_end, y_end)

        valid_points = (0 <= bresenham_path[0]) & (bresenham_path[0] < map_size) & (0 <= bresenham_path[1]) & (bresenham_path[1] < map_size)
        occupancy_map[bresenham_path[0, valid_points].astype(int), bresenham_path[1, valid_points].astype(int)] = -np.log(9)

        if 0 <= bresenham_path[0, -1] < map_size and 0 <= bresenham_path[1, -1] < map_size:
            occupancy_map[int(bresenham_path[0, -1]), int(bresenham_path[1, -1])] = np.log(18)

# Apply sigmoid function
sigmoid_map = expit(occupancy_map.T)

# Visualize the occupancy map
fig3 = plt.figure()
plt.imshow(sigmoid_map, origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.title('Comparison of Trajectories in Occupancy Grid map')
plt.plot(trajectory[0, :], trajectory[1, :], label='Odometry Trajectory')
plt.plot(T[0, 3, :], T[1, 3, :], label='ICP Optimised Trajectory')
plt.plot(T_GTSAM[0, 3, :], T_GTSAM[1, 3, :], label='GTSAM Optimized Trajectory')
plt.legend()
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.savefig('GTSAM Occupancy Grid Map')

################################################################################################################################

# extracting the pose (world-->robot) closest to the RGB image timestamp
comparison_matrix_encoder = encoder_stamps <= rgb_stamps[:,np.newaxis]
indices_encoder = np.argmax(~comparison_matrix_encoder, axis=1) - 1
T_world_robot = T_GTSAM[:, :, indices_encoder]

# extracting the disparity timestamps closest to the RGB image timestamp
disp_ts=disp_stamps[None]
rgb_ts=rgb_stamps[None].T
dt=abs(disp_ts-rgb_ts)
indices_disp = np.argmin(dt, axis=1)[:, np.newaxis]


def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

MAP1 = {}
MAP1['res']   = 0.05 #meters
MAP1['xmin']  = -25  #meters
MAP1['ymin']  = -25
MAP1['xmax']  =  25
MAP1['ymax']  =  25 
MAP1['sizex']  = int(np.ceil((MAP1['xmax'] - MAP1['xmin']) / MAP1['res'] + 1)) #cells
MAP1['sizey']  = int(np.ceil((MAP1['ymax'] - MAP1['ymin']) / MAP1['res'] + 1))
MAP1['map'] = np.zeros((MAP1['sizex'],MAP1['sizey'],3),dtype=np.int16) #DATA TYPE: char or int8

for i in range(indices_encoder.shape[0]-1):
    j = int(indices_disp[i])
    print('Texture Grid Map Itr No.:', i)
    # load RGBD image
    # imd = cv2.imread(disp_path+'disparity20_1.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imd = cv2.imread(rf'/mnt/c/Users/ITSloaner/Desktop/Khagesh/ECE 276A/ECE276A_PR2/data/KinectData/dataRGBD/Disparity{dataset}/disparity{dataset}_{j}.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    # imc = cv2.imread(rgb_path+'rgb20_1.png')[...,::-1] # (480 x 640 x 3)
    imc = cv2.imread(rf'/mnt/c/Users/ITSloaner/Desktop/Khagesh/ECE 276A/ECE276A_PR2/data/KinectData/dataRGBD/RGB{dataset}/rgb{dataset}_{i+1}.png')[...,::-1]

    # imd=cv2.imread(rf'/mnt/c/Users/saket/OneDrive - UC San Diego/Documents/Lectures/ECE 276(Sensing and estimation)/ECE276A_PR2/ECE276A_PR2/dataRGBD/Disparity{dataset}/Disparity{dataset}_{j}.png',cv2.IMREAD_UNCHANGED)
    # imc=cv2.imread(rf'/mnt/c/Users/saket/OneDrive - UC San Diego/Documents/Lectures/ECE 276(Sensing and estimation)/ECE276A_PR2/ECE276A_PR2/dataRGBD/RGB{dataset}/rgb{dataset}_{i+1}.png')[...,::-1]
    
    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

    # Transform optical frame coordinates to regular camera frame coordinates
    x_reg_cam = z[valid]
    y_reg_cam = -x[valid]
    z_reg_cam = -y[valid]
    clm_ones = np.ones((x_reg_cam.shape[0]))

    # convert position in the map frame here 
    Y_reg_cam = np.stack((x_reg_cam,y_reg_cam,z_reg_cam,clm_ones), axis =1)

    # Transformation matrix world frame to camera frame
    rotation_robot_cam = rotation_matrix_xyz(0, 0.36, 0.021)
    position_robot_cam = np.array([0.18, 0.005, 0.36])

    T_robot_cam = np.eye(4)
    T_robot_cam[:3,:3] = rotation_robot_cam
    T_robot_cam[:3,3] = position_robot_cam

    T_world_cam = T_world_robot[:,:,i] @ T_robot_cam

    # transformation to the world frame (rotated)
    Y_world_cam = Y_reg_cam @ T_world_cam.T

    worldcam_t = Y_world_cam.T
    indices_ext = worldcam_t[2, :] < -0.1
    Y_world_cam_clipped = worldcam_t[:, indices_ext]

    c1 = imc[rgbv[valid].astype(int), rgbu[valid].astype(int)]
    c2 = c1[indices_ext]

    x_texture_cell = (np.ceil((Y_world_cam_clipped[0,:] - MAP1['xmin']) / MAP1['res'] ).astype(np.int16))
    y_texture_cell=(np.ceil((Y_world_cam_clipped[1,:] - MAP1['ymin']) / MAP1['res'] ).astype(np.int16))
    indGood = np.logical_and(x_texture_cell < MAP1['sizex'], y_texture_cell < MAP1['sizey'])

    MAP1['map'][x_texture_cell[indGood], y_texture_cell[indGood]] = c2[indGood]
    map_array = MAP1['map']

textured_map = np.rot90(MAP1['map'], k=1)

fig3 = plt.figure()
plt.imshow(textured_map)
plt.plot(((trajectory[0, :]- MAP1['xmin']) / MAP1['res'] ), -((trajectory[1, :]- MAP1['ymin']) / MAP1['res'])+1000, label='Odometry Trajectory')
plt.plot(((T[0, 3, :]- MAP1['xmin']) / MAP1['res'] ), -((T[1, 3, :]- MAP1['ymin']) / MAP1['res'])+1000, label='ICP Optimised Trajectory')
plt.plot(((T_GTSAM[0, 3, :]- MAP1['xmin']) / MAP1['res'] ), -((T_GTSAM[1, 3, :]- MAP1['ymin']) / MAP1['res'])+1000, label='GTSAM Optimized Trajectory')
plt.legend()
plt.title('Comparison of Trajectories in Texture Grid map')
plt.xlabel('X (directional-cells)')
plt.ylabel('Y (directional-cells)')
plt.savefig('GTSAM Texture Grid Map')