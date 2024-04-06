import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree

lidar_pc = np.load('lidar_pc.npy')
trajectory = np.load('trajectory.npy')

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

def factorgraph_icp(k,j):
    # for i in range(trajectory.shape[1]-1):
    num_states = trajectory.shape[1]

    # transformation matric at t=0
    T_relative = np.zeros((4, 4, num_states - 1))
    T = np.zeros((4, 4, num_states))
    T[:,:,0] = np.eye(4)

    target_pc = lidar_pc[:,:,j]
    source_pc = lidar_pc[:,:,k]

    # transform matriax at ith time
    yaw = trajectory[2,j]
    translation = np.array([trajectory[0,j], trajectory[1,j], 0])
    T_t = transformation_matrix(yaw, translation)

    # transform matriax at (i+1)th time
    yaw = trajectory[2,k]
    translation = np.array([trajectory[0,k], trajectory[1,k], 0])
    T_t1 = transformation_matrix(yaw, translation)

    T_t_inv = np.linalg.inv(T_t)

    T_odometry = T_t_inv @ T_t1

    # calculate relative transformation using icp algorithm
    T_relative = icp_algorithm(T_odometry, source_pc, target_pc)

    return T_relative

