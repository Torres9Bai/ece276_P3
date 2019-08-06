# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:14:19 2019

@author: User
"""

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
    R=np.dot(R_roll, np.dot(R_yaw, R_pich))
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
    Rti=Euler_angle(Kante[0], Kante[1], Kante[2])
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


LM=0
LMU=np.vstack((np.zeros([3, features.shape[1]]), np.ones([1,features.shape[1]])))
sigma=1*np.eye(3)
D=np.vstack((np.eye(3), np.zeros([1,3])))
V=1000


while LM<features.shape[1]:
    for j in range (0, Luiz-1):#find the first picture including this landmark
        Pic=features[:,:,j]
        if Pic[0, LM]!=-1:
            zt0=Pic[:, LM]#.reshape(4,1)
            A=np.array([[f_su, 0, c_u-zt0[0]], [0, f_sv, c_v-zt0[1]], [f_su, 0, c_u-zt0[2]]])
            B=np.array([[0], [0], [f_su*b]])
            xx=solve(A, B)
            #RtT=(Rt[:, :, j]).T
            #mp=np.linalg.inv(Roc @ RtT) @ xx
            #mp=solve(Roc @ (Rt[:, :, j]).T, xx)
            #Barkley=TT[:, :, j]
            #p=Barkley[0:3, -1].reshape(3,1)#camera position in world 
            xx=np.vstack((xx, np.ones([1,1])))
            Barkley=np.linalg.inv(cam_T_imu) @ xx
            #L_miut0=mp+p#landmark position in the world
            L_miut0=TT[:, :, j] @ Barkley
            LMU[:, LM]=L_miut0.reshape(4)
            sigmat=sigma
            ss=j
            #print(L_miut0)
            break
    L_miut=L_miut0
    for k in range (ss, Luiz-1):#update the landmark
        Pict=features[:,:,k]
        if Pict[0, LM]!=-1:
            #L_miut=L_miut0
            zt_hat=M @ pimap(cam_T_imu @ TT_inv[:,:,k] @ L_miut)
            zt=Pict[:, LM]
            Ht=M @ dpi(cam_T_imu @ TT_inv[:,:,k] @ L_miut) @ cam_T_imu @ TT_inv[:,:,k] @ D
            Kt=sigmat @ Ht.T @ np.linalg.inv(Ht @ sigmat @ Ht.T + V*np.eye(4))
            L_miut1=L_miut+(D @ Kt @ (zt.reshape(4,1)-zt_hat)).reshape(4,1)
            LMU[:, LM]=L_miut1[:,0]#.reshape(4)
            sigmat=(np.eye(3) - Kt @ Ht) @ sigmat
            L_miut=L_miut1
    
    
    LM=LM+1
    #print(LM)

Sarriout=LMU
visualize_trajectory_2d(TT,LMU,show_ori=True)


