import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from functools import reduce
from tqdm import tqdm

import pykitti

basedir = '/mnt/Inside/KITTI/raw'

date = '2011_09_26'
drive = '0048'

dataset = pykitti.raw(basedir, date, drive)

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

# Grab some data
# second_pose = dataset.oxts[1].T_w_imu
# first_gray = next(iter(dataset.gray))
# first_cam1 = next(iter(dataset.cam1))
# first_rgb = dataset.get_rgb(0)
first_velo = dataset.get_velo(18)

# Display some of the data
# np.set_printoptions(precision=12, suppress=False)


# print('\nDrive: ' + str(dataset.drive))
# print('\nFrame range: ' + str(dataset.frames))
#
# print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
# print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
# print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))
#
# print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
# print('\nSecond IMU pose:\n' + str(second_pose))

# f, ax = plt.subplots(2, 2, figsize=(15, 5))
# ax[0, 0].imshow(first_gray[0], cmap='gray')
# ax[0, 0].set_title('Left Gray Image (cam0)')
#
# ax[0, 1].imshow(first_cam1, cmap='gray')
# ax[0, 1].set_title('Right Gray Image (cam1)')
#
# ax[1, 0].imshow(first_cam2)
# ax[1, 0].set_title('Left RGB Image (cam2)')
#
# ax[1, 1].imshow(first_rgb[1])
# ax[1, 1].set_title('Right RGB Image (cam3)')

f2 = plt.figure(figsize=(5, 10))
# ax2 = f2.add_subplot(111, projection='3d')
ax = f2.add_subplot(111)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 50)
ax.set_aspect('equal')
velo_range = range(0, first_velo.shape[0], 5)
ax.scatter(first_velo[velo_range, 1],
           first_velo[velo_range, 0],
           # first_velo[velo_range, 2],
           c=first_velo[velo_range, 3],
           s=0.1,
           cmap='rainbow')
ax.set_title('Velodyne scan (subsampled)')

plt.savefig('test.png', dpi=150, bbox_inches='tight')
