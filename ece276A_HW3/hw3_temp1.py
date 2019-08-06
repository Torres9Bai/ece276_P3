import numpy as np
from scipy.linalg import solve, block_diag
from utils import *
import math
from matplotlib import pyplot as plt
import cv2
import skimage.io as io
from numpy.random import uniform,randn,random
from scipy.linalg import expm, sinm, cosm


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM (Extra Credit)

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)


def I2W(x,y,z,roll,pich,yaw):
    R_yaw=np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_pich=np.array([[math.cos(pich), 0, math.sin(pich)], [0, 1, 0], [-math.sin(pich), 0, math.cos(pich)]])
    R_roll=np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R=np.dot(np.dot(R_yaw, R_pich), R_roll)
    #T=np.array([[R, np.array([[x],[y],[z]])], np.array([0, 0, 0, 1])])
    T=np.hstack((R, np.array([[x],[y],[z]])))
    T=np.vstack((T, np.array([0, 0, 0, 1])))
    return T

def hatmap(w1, w2, w3):
    w_hat=np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    return w_hat
'''
def vI2W(x, y, z, w, v):
    w_hat=hatmap(w[0,0], w[1,0], w[2,0])
    sw=np.array([[x], [y], [z]])
    sw_dot=np.dot(w_hat, sw)+v
    return sw_dot
'''

def Euler_angle(roll,pich,yaw):
    R_yaw=np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_pich=np.array([[math.cos(pich), 0, math.sin(pich)], [0, 1, 0], [-math.sin(pich), 0, math.cos(pich)]])
    R_roll=np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R=np.dot(np.dot(R_yaw, R_pich), R_roll)
    return R



dt0=t[0,1]-t[0,0]
Eden=rotational_velocity[:,0]*dt0
Hazard=linear_velocity[:,0]*dt0
Rt0=Euler_angle(Eden[0], Eden[1], Eden[2])
T0=I2W(Hazard[0], Hazard[1], Hazard[2], Eden[0], Eden[1], Eden[2])
Luiz=np.size(t)


T_iw_inv=np.linalg.inv(T0)
T_iw=T0
miu_t=T_iw_inv

TT=np.zeros([4,4,Luiz-1])
TT[:,:,0]=T0
Rt=np.zeros([3,3,Luiz-1])
Rt[:,:,0]=Rt0
TT_inv=np.zeros([4,4,Luiz-1])
TT_inv[:,:,0]=T_iw_inv

for i in range (1,Luiz-1):
    dt=t[0,i+1]-t[0,i]
    vt=linear_velocity[:,i]
    vt=vt.reshape(3,1)
    wt=rotational_velocity[:,i]
    Kante=wt*dt
    Rti=Euler_angle(Kante[0], Kante[1],Kante[2])
    Rt[:,:,i]=Rti
    Pedro=np.hstack((hatmap(wt[0], wt[1], wt[2]), vt))
    ut_hat=np.vstack((Pedro, np.zeros([1,4])))
    miu_t1=expm(-dt*ut_hat) @ miu_t
    #Willian=np.eye(4,4)+(-dt*ut_hat)+1/2*(-dt*ut_hat)@(-dt*ut_hat)
    #miu_t1=Willian @ miu_t
    miu_t=miu_t1
    TT_inv[:,:,i]=miu_t1
    Ti=np.linalg.inv(miu_t1)
    TT[:,:,i]=Ti
    
def pimap(q):
    aa=1/q[2, 0]*q
    return aa

def dpi(q):
    #Alonso=1/q[2, 0]*np.array([[1, 0, -q[0, 0]/q[2, 0], 0], [0, 1, -q[1, 0]/q[2, 0], 0], [0, 0, 0, 0], [0, 0, -q[3, 0]/q[2, 0], 1]])
    Alonso=1/q[2]*np.array([[1, 0, -q[0]/q[2], 0], [0, 1, -q[1]/q[2], 0], [0, 0, 0, 0], [0, 0, -q[3]/q[2], 1]])
    return Alonso

f_su=K[0,0]
c_u=K[0,2]
f_sv=K[1,1]
c_v=K[1,2]
M=np.array([[f_su, 0, c_u, 0], [0, f_sv, c_v, 0], [f_su, 0, c_u, -f_su*b], [0, f_sv, c_v, 0]])
Roc=np.array([[0,-1,0], [0, 0, -1], [1, 0, 0]])
M_pinv=np.linalg.pinv(M)

Pic0=features[:,:,0]
z0=np.zeros([4,1])
for i in range (Pic0.shape[1]):
    if Pic0[0,i]!=-1:
        z0=np.hstack((z0, Pic0[:, i].reshape(4,1)))

z0=z0[:, 1:]
xx=np.zeros([3,1])
for i in range (z0.shape[1]):
    Barkley=z0[:,i]
    A=np.array([[f_su, 0, c_u-Barkley[0]], [0, f_sv, c_v-Barkley[1]], [f_su, 0, c_u-Barkley[2]]])
    B=np.array([[0], [0], [f_su*b]])
    xxi=solve(A, B)
    xx=np.hstack((xx,xxi))

xx=xx[:, 1:]
mp0=np.linalg.inv(Roc@Rt0)@xx
p0=T0[0:3,-1].reshape(3,1)
L_miut=mp0-p0
L_miut=np.vstack((L_miut, np.ones([1, z0.shape[1]])))

L_miu=np.zeros([4,z0.shape[1],Luiz-1])
L_miu[:,:,0]=L_miut

#sigmat=0.1*np.eye(3*z0.shape[1])
sigmat=0.1*np.eye(3)

D=np.vstack((np.eye(3), np.zeros([1,3])))
V=30
for i in range (0, 1):
    Pic=features[:,:,i]
    zt=np.zeros([4,1])
    for i in range (Pic.shape[1]):
        if Pic0[0,i]!=-1:
            zt=np.hstack((zt, Pic[:, i].reshape(4,1)))
    zt=zt[:,1:]
    zt_hat=np.zeros([4, zt.shape[1]])
    for j in range (zt.shape[1]):
        miu_tj=L_miut[:, j].reshape(4,1)
        zt_hatj=M @ pimap(cam_T_imu @ TT_inv[:,:,i] @ miu_tj)
        zt_hat[:,j]=zt_hatj.reshape(4)
    #H_t=M @ dpi(cam_T_imu @ TT_inv[:,:,i] @ L_miut[:, 0]) @ cam_T_imu @ TT_inv[:,:,i] @ D
    for k in range (0, zt.shape[1]):
        #H_tt=M @ dpi((cam_T_imu @ TT_inv[:,:,i] @ L_miut[:, k]).reshape(4,1)) @ cam_T_imu @ TT_inv[:,:,i] @ D
        H_t=M @ dpi(cam_T_imu @ TT_inv[:,:,i] @ L_miut[:, k]) @ cam_T_imu @ TT_inv[:,:,i] @ D
        #H_t=block_diag(H_t, H_tt)
        #D=block_diag(D, D)
        Kt=sigmat @ H_t.T @ np.linalg.inv(H_t @ sigmat @ H_t.T + V*np.eye(zt.shape[1]))
    #D=block_diag(D, D, D, D)
        L_miut1=L_miut+D @ Kt @ (zt-zt_hat)
    sigmat=(np.eye(3*zt.shape[1]) - Kt @ H_t) @ sigmat
    #sigmat=np.dot((np.eye(3*zt.shape[1] - Kt @ H_t), sigmat)
    L_miu[:,:]=L_miut1
    L_miut=L_miut1
    print(i)




    

