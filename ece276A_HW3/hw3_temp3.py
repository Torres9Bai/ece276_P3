# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:08:03 2019

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

def Euler_angle(roll,pich,yaw):
    R_yaw=np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_pich=np.array([[math.cos(pich), 0, math.sin(pich)], [0, 1, 0], [-math.sin(pich), 0, math.cos(pich)]])
    R_roll=np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R=np.dot(np.dot(R_yaw, R_pich), R_roll)
    return R

def pimap(q):
    aa=1/q[2, 0]*q
    return aa

def dpi(q):
    #Alonso=1/q[2, 0]*np.array([[1, 0, -q[0, 0]/q[2, 0], 0], [0, 1, -q[1, 0]/q[2, 0], 0], [0, 0, 0, 0], [0, 0, -q[3, 0]/q[2, 0], 1]])
    Alonso=1/q[2]*np.array([[1, 0, -q[0]/q[2], 0], [0, 1, -q[1]/q[2], 0], [0, 0, 0, 0], [0, 0, -q[3]/q[2], 1]])
    return Alonso

def hhatmap(v, w):
    Mane=np.array([[hatmap(w[0,0], w[0,1], w[0,2]), hatmap(v[0,0], v[0,1], v[0,2])], [np.zeros([3,3]), hatmap(w[0,0], w[0,1], w[0,2])]])
    return Mane



f_su=K[0,0]
c_u=K[0,2]
f_sv=K[1,1]
c_v=K[1,2]
M=np.array([[f_su, 0, c_u, 0], [0, f_sv, c_v, 0], [f_su, 0, c_u, -f_su*b], [0, f_sv, c_v, 0]])
Roc=np.array([[0,-1,0], [0, 0, -1], [1, 0, 0]])

dt0=t[0,1]-t[0,0]
Eden=rotational_velocity[:,0]*dt0
Hazard=linear_velocity[:,0]*dt0
Rt0=Euler_angle(Eden[0], Eden[1], Eden[2])
T0=I2W(Hazard[0], Hazard[1], Hazard[2], Eden[0], Eden[1], Eden[2])

Luiz=np.size(t)


T_iw_inv=np.linalg.inv(T0)
T_iw=T0
miu_t=T_iw_inv

Ppt=np.zeros([4,4,Luiz-1])
Ppt[:,:,0]=T0
Rt=np.zeros([3,3,Luiz-1])
Rt[:,:,0]=Rt0
Ppt_inv=np.zeros([4,4,Luiz-1])
Ppt_inv[:,:,0]=T_iw_inv

LM=features.shape[1]

Plt=np.zeros([4, LM])

Pic0=features[:,:,0]
for i in range (Pic0.shape[1]):
    if Pic0[0, i]!=-1:
        ii=i
        zt0=Pic0[:, ii]#.reshape(4,1)
        A=np.array([[f_su, 0, c_u-zt0[0]], [0, f_sv, c_v-zt0[1]], [f_su, 0, c_u-zt0[2]]])
        B=np.array([[0], [0], [f_su*b]])
        xx=solve(A, B)
        xx=np.vstack((xx, np.ones([1,1])))
        Barkley=np.linalg.inv(cam_T_imu) @ xx
        Plt0=Ppt[:, :, 0] @ Barkley
        Plt[:, ii]=Plt0[:, 0]


Miu=np.zeros([4,4+LM])
Miu[:,0:4]=Ppt_inv[:, :, 0]
Miu[:,4:]=Plt
sigmat=np.eye(6+3*LM)
W=np.eye(6)

for i in range (1,Luiz-1):
    dt=t[0,i+1]-t[0,i]
    vt=linear_velocity[:,i]
    vt=vt.reshape(3,1)
    wt=rotational_velocity[:,i]
    Pedro=np.hstack((hatmap(wt[0], wt[1], wt[2]), vt))
    ut_hat=np.vstack((Pedro, np.zeros([1,4])))
    Ppti=expm(-dt*ut_hat) @ Miu[:,0:4]
    Plti=Miu[:,4:]
    ut_hhat=hhatmap(vt, wt)
    Higuain=expm(-dt*ut_hhat)
    Psigma=block_diag(Higuain, np.eye(3*LM)) @ sigmat @ block_diag(Higuain.T, np.eye(3*LM)) + dt*W
    
    

        



