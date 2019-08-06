
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import signal

def d_pi_d_q(q):
    q1,q2,q3,q4 = q
    result = (1/q3)*np.array([[1,0,-q1/q3, 0],
                            [0,1,-q2/q3,0],
                            [0,0,0,0],
                            [0,0,-q4/q3,1]])
    return result

def pi_(q):
    q1,q2,q3,q4 = q
    return q/q3

def circle_point(vec):
    result = np.zeros((4, 6))
    s= vec[0:-1]
    lam = vec[-1]
    result[0:3,0:3] = lam*np.eye(3)
    result[0:3,3:] =  -skew_so(s)
    return result

def skew_so(theta):
    G_x = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G_y = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    alpha_1, alpha_2, alpha_3 = theta
    return alpha_1*G_x+alpha_2*G_y+alpha_3*G_z

def skew_se(xi):
    rou = xi[0:3]
    theta = xi[3:]
    theta_hat = skew_so(theta)
    result = np.zeros((4,4))
    result[0:3,0:3] = theta_hat
    result[0:3,-1] = rou.reshape(3,)
    return result

def vee(xi):
    result = np.zeros((6, 6))
    rou = xi[0:3]
    theta = xi[3:]
    theta_hat = skew_so(theta)
    rou_hat = skew_so(rou)
    result[0:3, 0:3] = theta_hat
    result[0:3, 3:] = rou_hat
    result[3:, 3:] = theta_hat
    return result

def expm_vee(xi_vee,xi):
    theta = xi[3:]
    theta_norm = np.linalg.norm(theta)
    return np.eye(6)+(3*np.sin(theta_norm)-theta_norm*np.cos(theta_norm))/(2*theta_norm)*xi_vee+\
    (4-theta_norm*np.sin(theta_norm)-4*np.cos(theta_norm))/(2*theta_norm**2)*xi_vee**2+\
    (np.sin(theta_norm)-theta_norm*np.cos(theta_norm))/(2*theta_norm**3)*xi_vee**3+\
    (2-theta_norm*np.sin(theta_norm)-2*np.cos(theta_norm))/(2*theta_norm**4)*xi_vee**4


def prediction(xi,T,robot_sigma,robot_poses, time, delta_time):
    u_hat = skew_se(xi)
    u_vee = vee(xi)
    T[:,:,time] = expm(-u_hat).dot(T[:,:,time-1])
    W = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])*1
    robot_sigma = expm_vee(-u_vee,xi).dot(robot_sigma).dot(expm_vee(-u_vee,xi).T) \
            + W * delta_time ** 2
    robot_poses[:,:,time] = np.linalg.inv(T[:,:,time])
    return T, robot_sigma,robot_poses

def opt2world(coor, T, M):
    R_oc = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    R = np.linalg.inv(T)[0:3, 0:3]
    p = np.linalg.inv(T)[0:3, -1]
    u_L, v_L, u_R, v_R = coor
    z = -M[2, 3] / (u_L - u_R)
    x = (coor[0] - M[0, 2]) * z / M[0, 0]
    y = (coor[1] - M[1, 2]) * z / M[1, 1]
    temp = np.array([x, y, z, 1])
    coor[2]=0
    coor[0:3] = np.linalg.inv(R_oc.dot(R.T)).dot(temp[0:3]) + p
    coor[3]=1
    return coor

def Mapping(T, cam_T_imu,M, landmark_mu, landmark_sigma, features,time,D_):
    m = features.shape[1]
    feature = features[:,:,time]
    ind_Good = (feature[0, :] != -1)
    N_t = np.sum(ind_Good)
    landmark_H = np.zeros((4*N_t,3*m))
    z_hat = np.zeros((4,N_t))
    inds = np.array([i for i in range(m) if ind_Good[i]])
    D = np.vstack((np.eye(3 * m), np.zeros((m, 3 * m))))
    if N_t==0:
        return landmark_mu
    else:
        for i in range(N_t):
            if inds[i] not in initialized:
                landmark_mu[:,inds[i]] = opt2world(feature[:,inds[i]],T[:,:,time],M)
                initialized.append(inds[i])
                continue
            else:
                z_hat[:,i] = M.dot(pi_(cam_T_imu.dot(T[:,:,time]).dot(landmark_mu[:,inds[i]])))
                landmark_H[4*i:4*i+4,3*inds[i]:3*inds[i]+3] = \
                    M.dot(d_pi_d_q(cam_T_imu.dot(T[:,:,time]).dot(landmark_mu[:,inds[i]]))).dot(cam_T_imu).dot(T[:,:,time]).dot(D_)
        landmark_K = landmark_sigma.dot(landmark_H.T).dot(np.linalg.pinv(landmark_H.dot(landmark_sigma).dot(landmark_H.T)+np.eye(4*N_t)*10))
        landmark_mu =  (landmark_mu.reshape(-1,1)+(D.dot(landmark_K).dot((feature[:,ind_Good]-z_hat).T.reshape(4*N_t,1)))).reshape(4,-1)
        landmark_sigma = (np.eye(3*m)-landmark_K.dot(landmark_H)).dot(landmark_sigma)
    return landmark_mu


def update(T, sigma, M, cam_T_imu, landmark_mu, features, robot_poses,time,D_):
    m = features.shape[1]
    feature = features[:, :, time]
    ind_Good = (feature[0, :] != -1)
    N_t = np.sum(ind_Good)
    landmark_H = np.zeros((4 * N_t, 3 * m))
    z_hat = np.zeros((4, N_t))
    inds = np.array([i for i in range(m) if ind_Good[i]])
    D = np.vstack((np.eye(3 * m), np.zeros((m, 3 * m))))
    H = np.zeros((4*N_t,3*m+6))
    if N_t == 0:
        return T, sigma, robot_poses,landmark_mu
    else:
        for i in range(N_t):
            if inds[i] not in initialized:
                landmark_mu[:, inds[i]] = opt2world(feature[:, inds[i]], T[:, :, time], M)
                initialized.append(inds[i])
                continue
            else:
                z_hat[:, i] = M.dot(pi_(cam_T_imu.dot(T[:, :, time]).dot(landmark_mu[:, inds[i]])))
                landmark_H[4 * i:4 * i + 4, 3 * inds[i]:3 * inds[i] + 3] = \
                    M.dot(d_pi_d_q(cam_T_imu.dot(T[:, :, time]).dot(landmark_mu[:, inds[i]]))).dot(cam_T_imu).dot(
                        T[:, :, time]).dot(D_)
                # H[:,:3*m] = M.dot(d_pi_d_q(cam_T_imu.dot(T[:, :, time]).dot(landmark_mu[:, inds[i]]))).dot(cam_T_imu).dot(
                #         T[:, :, time]).dot(D_)
                H[4*i:4*i+4,3*m:3*m+6] = M.dot(d_pi_d_q(cam_T_imu.dot(T[:, :, time]).dot(landmark_mu[:, inds[i]]))).dot(
                    cam_T_imu).dot(circle_point(T[:, :, time].dot(landmark_mu[:, inds[i]])))
        H[:,:3*m] = landmark_H
        # H = np.hstack((landmark_H, np.repeat(robot_H, N_t, axis=0)))
        K = sigma.dot(H.T).dot(np.linalg.pinv(H.dot(sigma).dot(H.T) + np.eye(4 * N_t) * 2.5))
        landmark_mu = (landmark_mu.reshape(-1, 1) + (
            D.dot(K[:3*m,:]).dot((feature[:, ind_Good] - z_hat).T.reshape(4 * N_t, 1)))).reshape(4, -1)
        T[:,:,time] = expm(skew_se(K[3*m:3*m+6,:].dot((feature[:,ind_Good]-z_hat).reshape(4*N_t,1)))).dot(T[:,:,time])
        sigma = (np.eye(3 * m + 6) - K.dot(H)).dot(sigma)
        robot_poses[:, :, time - 1] = np.linalg.inv(T[:, :, time])
    return T, sigma, robot_poses,landmark_mu



if __name__ == '__main__':
    file = "27"
    filename = "./data/00"+file+".npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    m = features.shape[1]
    M = np.zeros((4,4))
    M[0:2, 0:3] = K[0:2, :]
    M[2:4, 0:3] = K[0:2, :]
    M[2, -1] = -K[0,0] * b
    start_time_ind = 0
    end_time_ind = t.shape[1]

    # (a) IMU Localization via EKF Prediction
    robot_poses = np.zeros((4, 4, end_time_ind - start_time_ind))
    T = np.zeros((4, 4, end_time_ind - start_time_ind))
    T[:, :, 0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    robot_sigma = np.eye(6)
    for time in range(start_time_ind + 1, end_time_ind):
        delta_time = t[:, time] - t[:, time - 1]
        omega = rotational_velocity[:, time]
        v = linear_velocity[:, time]
        xi = np.vstack((v.reshape(3, 1), omega.reshape(3, 1))) * delta_time
        T, robot_sigma,robot_poses = prediction(xi, T, robot_sigma,robot_poses, time, delta_time)
    visualize_trajectory_2d(robot_poses)

    # (b) Landmark Mapping via EKF Update
    m = features.shape[1]
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    landmark_mu = np.zeros((4,m))
    landmark_sigma = np.eye(3 * m)
    initialized = list()
    for time in range(start_time_ind + 1, end_time_ind):
        landmark_mu = Mapping(T, cam_T_imu, M, landmark_mu, landmark_sigma, features, time, D)
    filt = np.abs(landmark_mu-np.mean(landmark_mu))<800
    visualize_trajectory_2d(robot_poses, landmark_mu.reshape(4,m)[:,filt[0,:]], show_landmark_b=True)

    # (c) Visual-Inertial SLAM (Extra Credit)
    robot_poses_c = np.zeros((4, 4, end_time_ind - start_time_ind))
    T = np.zeros((4, 4, end_time_ind - start_time_ind))
    T[:, :, 0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    m = features.shape[1]
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    landmark_mu_c = np.zeros((4, m))
    sigma = np.eye(3 * m + 6) * 1e-10
    initialized = list()
    for time in range(start_time_ind + 1, end_time_ind):
        delta_time = t[:, time] - t[:, time - 1]
        omega = rotational_velocity[:, time]
        v = linear_velocity[:, time]
        xi = np.vstack((v.reshape(3, 1), omega.reshape(3, 1))) * delta_time
        T, robot_sigma, _ = prediction(xi, T, robot_sigma, robot_poses_c, time, delta_time)
        T, sigma, robot_poses_c,landmark_mu_c, = update(T, sigma, M, cam_T_imu, landmark_mu_c, features, robot_poses_c,time,D)
    filt = np.abs(landmark_mu - np.mean(landmark_mu)) < 800
    filt_c = np.abs(landmark_mu_c - np.mean(landmark_mu_c)) < 800
    # You can use the function below to visualize the robot pose over time
    visualize_trajectory_2d(robot_poses, landmark_mu[:,filt[0,:]], robot_poses_c, landmark_mu_c[:,filt_c[0,:]], show_landmark_b=True, show_pose2=True,show_landmark_c=True,)