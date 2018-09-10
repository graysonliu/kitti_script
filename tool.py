import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from functools import reduce

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
first_cam2 = dataset.get_cam2(0)
first_velo = dataset.get_velo(0)


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

# f2 = plt.figure()
# ax2 = f2.add_subplot(111, projection='3d')
# # Plot every 100th point so things don't get too bogged down
# velo_range = range(0, first_velo.shape[0], 100)
# ax2.scatter(first_velo[velo_range, 0],
#             first_velo[velo_range, 1],
#             first_velo[velo_range, 2],
#             c=first_velo[velo_range, 3],
#             cmap='rainbow')
# ax2.set_title('Third Velodyne scan (subsampled)')


# plt.show()


# for projecting point cloud to image
def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
       points can be directly multiplied by the camera projection matrix'''

    pts3d = pts3d_raw
    # Reflectance > 0
    # pts3d = pts3d[pts3d[:, 3] > 0, :]
    pts3d[:, 3] = 1
    return pts3d.transpose()


def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''

    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])

    return idx, pts2d_cam / pts2d_cam[2, :]


# generate testing folder
testing_dir = os.path.join(dataset.data_path, 'testing')
os.system("rm -rf {}".format(testing_dir))
image_dir = os.path.join(testing_dir, "image_2")
velodyne_dir = os.path.join(testing_dir, "velodyne")
calib_dir = os.path.join(testing_dir, "calib")
plane_dir = os.path.join(testing_dir, "planes")
list(map(os.mkdir, [testing_dir, image_dir, velodyne_dir, calib_dir, plane_dir]))

# copy image and velodyne
image_source_dir = os.path.join(dataset.data_path, 'image_02', 'data')
image_list = os.listdir(image_source_dir)
list(map(lambda x: os.system("cp {} {}".format(os.path.join(image_source_dir, x), os.path.join(image_dir, x[-10:]))),
         image_list))
velodyne_source_dir = os.path.join(dataset.data_path, 'velodyne_points', 'data')
velodyne_list = os.listdir(velodyne_source_dir)
list(map(
    lambda x: os.system("cp {} {}".format(os.path.join(velodyne_source_dir, x), os.path.join(velodyne_dir, x[-10:]))),
    velodyne_list))

# create test.txt, plane file and calib file
test_list = open(os.path.join(dataset.data_path, "test.txt"), 'w')
plane_str = "# Plane\n" + "Width 4\n" + "Height 1\n" + "0 -1 0 1.65\n"
for i in range(len(image_list)):
    test_list.write("%06d\n" % i)
    with open(os.path.join(plane_dir, "%06d.txt" % i), 'w') as f:
        f.write(plane_str)
    with open(os.path.join(calib_dir, "%06d.txt" % i), 'w') as f:
        p0 = reduce(lambda x, y: x + " %.12e" % y, dataset.calib.P_rect_00.reshape(-1).tolist(), "")
        p1 = reduce(lambda x, y: x + " %.12e" % y, dataset.calib.P_rect_10.reshape(-1).tolist(), "")
        p2 = reduce(lambda x, y: x + " %.12e" % y, dataset.calib.P_rect_20.reshape(-1).tolist(), "")
        p3 = reduce(lambda x, y: x + " %.12e" % y, dataset.calib.P_rect_30.reshape(-1).tolist(), "")
        r0_rect = reduce(lambda x, y: x + " %.12e" % y, dataset.calib.R_rect_00[0:3, 0:3].reshape(-1).tolist(), "")
        tr_velo_to_cam = reduce(lambda x, y: x + " %.12e" % y,
                                dataset.calib.T_cam0_velo_unrect[0:3].reshape(-1).tolist(), "")
        f.write("P0:{}\nP1:{}\nP2:{}\nP3:{}\nR0_rect:{}\nTr_velo_to_cam:{}".format(p0, p1, p2, p3, r0_rect,
                                                                                   tr_velo_to_cam))
test_list.close()

T_cam_velo = dataset.calib.T_cam0_velo_unrect
R_rect_00 = dataset.calib.R_rect_00
P_rect_02 = dataset.calib.P_rect_20

velo_range = range(0, first_velo.shape[0], 4)
first_velo = first_velo[velo_range, :]
reflectance = first_velo[:, 3].copy()
velo_prepared = prepare_velo_points(first_velo)
idx, b = project_velo_points_in_img(velo_prepared, T_cam_velo, R_rect_00, P_rect_02)

save_dir = os.path.join(basedir, date, "projection")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

width, height, dpi = 1242, 375, 96
fig = plt.figure()
fig.set_size_inches([width / dpi, height / dpi])
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_aspect('equal')
ax.imshow(first_cam2)
# ax.scatter(b[0, :], b[1, :], s=0.2, c=reflectance[idx], cmap='rainbow')
fig.savefig(os.path.join(save_dir, "kk"), dpi=dpi)
# plt.show()

# create calib file
print(dataset.calib.P_rect_20.reshape(-1))
print(dataset.calib.R_rect_00[0:3, 0:3].reshape(-1))
print(dataset.calib.T_cam0_velo_unrect[0:3].reshape(-1))
