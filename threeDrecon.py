import os
import cv2
from get_sift import *
from essential_mat import *
from trajectory import *
from plot_trajectory import *
from bag_of_words import *
from descriptors import *
import open3d as o3d

folder_path = 'C:/Users/bhave/PycharmProjects/SIR/Kitti_Seq_07/'

getsift = get_sift(folder_path)
list_kp1,list_kp2,list_dp1 = getsift.getsift()

K1 = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02],
               [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

K2 = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02],
               [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

get_essential = get_essential(list_kp1,list_kp2,K1,K2)
R_est,T_est = get_essential.essential_mat()

I = np.identity(3)
zeros = np.zeros((3, 1))
RT_zero = np.hstack((I,zeros))
RT_one = np.hstack((R_est[0],T_est[0]))

P_zero = np.matmul(K2,RT_zero)
P_one = np.matmul(K2,RT_one)

projpt0 = np.array(list_kp1[0])
projpt1 = np.array(list_kp2[0])
points = cv2.triangulatePoints(P_zero, P_one, projpt0.T, projpt1.T)
points4D = points.T

points3D = []
for pts in points4D:

    x = pts[0]/pts[3]
    y = pts[1]/pts[3]
    z = pts[2]/pts[3]

    points3D.append((x,y,z))

points3D[:] = np.asarray(points3D[:])

xyz = points3D.copy
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D)
o3d.visualization.draw_geometries([pcd])