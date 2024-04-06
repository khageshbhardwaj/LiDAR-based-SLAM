import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
import cv2
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time

dataset = 20

with np.load(f".\data\Encoders{dataset}.npz") as data:
  encoder_counts = data["counts"] # 4 x n encoder counts
  encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load(f".\data\Hokuyo{dataset}.npz") as data:
  lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
  lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
  lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
  lidar_range_min = data["range_min"] # minimum range value [m]
  lidar_range_max = data["range_max"] # maximum range value [m]
  lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
  lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load(f".\data\Imu{dataset}.npz") as data:
  imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
  imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
  imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load(f".\data\Kinect{dataset}.npz") as data:
  disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
  rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

def calculate_timestamps(abs_timestamps):
    # Ensure the input array is not empty
    if len(abs_timestamps) == 0:
        return []

    inc_timestamps = abs_timestamps - abs_timestamps[0]

    return inc_timestamps

# incremental timestamps
timestamp_encoder = calculate_timestamps(encoder_stamps)
timestamp_lidar = calculate_timestamps(lidar_stamps)
timestamp_imu = calculate_timestamps(imu_stamps)
timestamp_rgb = calculate_timestamps(rgb_stamps)

# encoder and IMU odometry
# distance travelled by the wheels
dist_r = (encoder_counts[0,:]+encoder_counts[2,:])/2*0.0022
dist_l = (encoder_counts[1,:]+encoder_counts[3,:])/2*0.0022

# velocity of the differential robot

def differential_data(data_series):
    # Ensure the input array is not empty
    if len(data_series) < 2:
        return []

    diff_data = []

    for i in range(len(data_series)-1):
        diff = data_series[i+1] - data_series[i]
        diff_data.append(diff)

    return np.array(diff_data)

# delta time t
dt_encoder = differential_data(timestamp_encoder)

# slice the distance travelled list to mathc the dt_encoder list
d_dist_r = dist_r[:-1]
d_dist_l = dist_l[:-1]

# Velocity
vr = d_dist_r/dt_encoder
vl = d_dist_l/dt_encoder

v = (vr+vl)/2

# extracting the IMU data closest to the encoder timestamp
comparison_matrix = imu_stamps <= encoder_stamps[:,np.newaxis]
indices = np.argmax(~comparison_matrix, axis=1) - 1

imu_omega = imu_angular_velocity[:, indices]
w = imu_omega[2,:]
print(imu_omega.shape)

def euler_discretization(x0, v, w, dt):
    # Update state using Euler discretization
    xt = np.zeros((3, len(v)+1))
    xt[:,0] = x0

    for i in range(len(v)):
        xt[:,i+1] = xt[:,i] + dt[i] * np.array([v[i] * np.cos(xt[2,i]), v[i] * np.sin(xt[2,i]), w[i]])

    return xt

# initialization
x0 = np.array([0,0,0])

trajectory = euler_discretization(x0, v, w, dt_encoder)
np.save('trajectory.npy', trajectory)

# Plot x vs y
plt.title('Trajectory in x-y Plane')
plt.plot(trajectory[0, :], trajectory[1, :], label='Odometry Trajectory')
plt.legend()
plt.title('Differential Drive Trajectory')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.grid(True)
plt.savefig('Odometry Trajectory')

###############################################################################################
# Part 2
###############################################################################################

def visualize_icp_result(source_pc, target_pc, pose):
    '''
    Visualize the result of ICP
    source_pc: numpy array, (N, 3)
    target_pc: numpy array, (N, 3)
    pose: SE(4) numpy array, (4, 4)
    '''
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
    source_pcd.paint_uniform_color([0, 0, 1])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
    target_pcd.paint_uniform_color([1, 0, 0])

    source_pcd.transform(pose)

    o3d.visualization.draw_geometries([source_pcd, target_pcd])

def transformation_matrix(theta, translation):

    # Create a rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Create a pose matrix with zero translation and rotation about the z-axis
    pose = np.eye(4)

    pose[:3, :3] = rotation_matrix[:3, :3]
    pose[:3, 3] = translation


    return pose

def kabsch_algorithm(Z, M):
    z_centroid = np.mean(Z, axis = 0)
    m_centroid = np.mean(M, axis = 0)

    Q = np.zeros((3,3))

    delta_m = M - m_centroid
    delta_z = Z - z_centroid

    # Q = sum(np.outer(dm, dz) for dm, dz in zip(delta_m, delta_z))
    Q = delta_m.T @ delta_z

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(Q)

    # Compute det(UV^T)
    det_UVt = np.linalg.det(U @ Vt)

    # Create the scaling matrix with ones and det(UV^T)
    scaling_matrix = np.eye(len(S))
    scaling_matrix[-1, -1] = det_UVt

    # Construct R using U and Vt
    R = U @ scaling_matrix @ Vt
    P = m_centroid - R @ z_centroid

    return R, P

def icp_algorithm(T0, Z, M):
    max_iteration = 10
    R = T0[:3, :3]
    P = T0[:3, 3]

    tree = KDTree(M)

    for i in range(max_iteration):
        Z_tilda = (R @ Z.T + P[:, np.newaxis]).T

        # D2 = np.sum(M_tilda[...,None,:] - Z[None, ...]**2, axis = 2)
        # jj = np.argmin(D2, axis =1)

        # Use KDTree query with specified number of neighbors (k=1)
        dd, jj = tree.query(Z_tilda)
        M_tilda = M[jj, :]

        R, P = kabsch_algorithm(Z, M_tilda)

    transformed_matrix = np.eye(4)
    transformed_matrix[:3,:3] = R
    transformed_matrix[:3,3] = P

    return transformed_matrix

# LiDAR data
num_points = 1081
field_of_view = 270  # degrees
angle_increment = field_of_view / (num_points - 1)

# Convert range measurements to (x, y) coordinates in sensor frame
angles = np.radians(np.linspace(-field_of_view / 2, field_of_view / 2, num_points))
ranges = lidar_ranges

x_lidar = np.zeros_like(ranges)
y_lidar = np.zeros_like(ranges)

for i in range(ranges.shape[1]):
    x_lidar[:,i] = (ranges[:,i] * np.cos(angles)) + 0.15
    y_lidar[:,i] = ranges[:,i] * np.sin(angles)

z_lidar = np.zeros((x_lidar.shape[0], x_lidar.shape[1]))

# Stack x, y, z arrays along the second axis
lidar_pc = np.stack((x_lidar, y_lidar, z_lidar), axis=1)

# Transpose the array to have shape (num_points, num_scans, 3)
# lidar_points = np.transpose(lidar_points, axes=(0, 2, 1))

np.save('lidar_pc', lidar_pc)

def transformation_inverse(T):
    # Extract rotation matrix R and translation vector P from T
    R = T[:3, :3]
    P = T[:3, 3]

    # Calculate the inverse of R
    R_inv = np.transpose(R)

    # Construct the inverse transformation matrix T_inv
    T_inv = np.zeros_like(T)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -np.dot(R_inv, P)
    T_inv[3, 3] = 1

    return T_inv

# for i in range(trajectory.shape[1]-1):
num_states = min(lidar_stamps.shape[0], encoder_stamps.shape[0])

# transformation matric at t=0
T_relative = np.zeros((4, 4, num_states - 1))
T = np.zeros((4, 4, num_states))
T[:,:,0] = np.eye(4)

for i in range(num_states-1):
    print("ICP Iteration:", i)
    target_pc = lidar_pc[:,:,i]
    source_pc = lidar_pc[:,:,i+1]

    # transform matriax at (i+1)th time
    yaw = trajectory[2,i+1]
    translation = np.array([trajectory[0,i+1], trajectory[1,i+1], 0])
    T_t1 = transformation_matrix(yaw, translation)

    # transform matriax at ith time
    yaw = trajectory[2,i]
    translation = np.array([trajectory[0,i], trajectory[1,i], 0])
    T_t = transformation_matrix(yaw, translation)

    T_t_inv = np.linalg.inv(T_t)

    T_odometry = T_t_inv @ T_t1

    # calculate relative transformation using icp algorithm
    T_relative[:,:,i] = icp_algorithm(T_odometry, source_pc, target_pc)

    if np.linalg.norm(T_relative [:,:,i][:3,:3] - T_odometry[:3,:3], 'fro')> 0.01:
        T_relative[:,:,i] = T_odometry

    # transformations during all states
    T[:,:,i+1] = T[:,:,i] @ T_relative[:,:,i]

    # # visualize the estimated result
    # visualize_icp_result(source_pc, target_pc, T_relative[:, :, i])

# Plot x vs y
fig3 = plt.figure()
plt.title('Trajectory in x-y Plane')
plt.plot(trajectory[0, :], trajectory[1, :], label='Odometry Trajectory')
plt.plot(T[0, 3, :], T[1, 3, :], label='ICP Optimised Trajectory')
plt.legend()
plt.title('Comparison of Trajectories')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.grid(True)
plt.savefig('ICP Optimised Trajectory')

# Save pose matrix to a .npy file
np.save("pose_matrix_T_relative.npy", T_relative)
np.save("pose_matrix_T.npy", T)

##########################################################################################################################
# Part 3
##########################################################################################################################

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT
  im              the map
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)
  xs,ys           physical x,y,positions you want to evaluate "correlation"

  OUTPUT
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr

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

# Initialize the occupancy grid map
map_resolution = 0.05  # meters per cell
map_size = 1000  # size of the map in cells
xmin = -25
xmax = 25
ymin = -25
ymax = 25
occupancy_map = np.zeros((map_size, map_size), dtype=np.float32)

for i in range(num_states-1):
    print('Occupancy Map Iteration:', i)
    clm_ones = np.ones((x_lidar.shape[0]))

    # convert position in the map frame here
    Yt = np.stack((x_lidar[:,i],y_lidar[:,i],z_lidar[:,i],clm_ones), axis =1)

    # transformation to the world frame (rotated)
    T_pose = np.load('pose_matrix_T.npy')
    Yr = Yt @ T_pose[:,:,i].T

    xs0 = Yr[:,0]
    ys0 = Yr[:,1]

    x_start = int((T_pose[:,:,i][0,3] - xmin) / map_resolution)
    y_start = int((T_pose[:,:,i][1,3] - ymin) / map_resolution)

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
plt.legend()
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.savefig('Occupancy Grid Map')

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

# extracting the pose (world-->robot) closest to the RGB image timestamp
comparison_matrix_encoder = encoder_stamps <= rgb_stamps[:,np.newaxis]
indices_encoder = np.argmax(~comparison_matrix_encoder, axis=1) - 1
T_world_robot = T[:, :, indices_encoder]

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
    print('Texture Map Iteration:', i)
    # load RGBD image
    # imd = cv2.imread(disp_path+'disparity20_1.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imd = cv2.imread(rf'C:\Users\ITSloaner\Desktop\Khagesh\ECE 276A\ECE276A_PR2\data\KinectData\dataRGBD\Disparity{dataset}\disparity{dataset}_{j}.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    # imc = cv2.imread(rgb_path+'rgb20_1.png')[...,::-1] # (480 x 640 x 3)
    imc = cv2.imread(rf'C:\Users\ITSloaner\Desktop\Khagesh\ECE 276A\ECE276A_PR2\data\KinectData\dataRGBD\RGB{dataset}\rgb{dataset}_{i+1}.png')[...,::-1]

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
plt.legend()
plt.title('Comparison of Trajectories in Texture Grid map')
plt.xlabel('X (directional-cells)')
plt.ylabel('Y (directional-cells)')
plt.savefig('Texture Grid Map')