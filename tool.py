import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from functools import reduce
from tqdm import tqdm

import pykitti

basedir = '/mnt/Inside/KITTI/raw'

date = '2011_09_26'
drive = '0048'

dataset = pykitti.raw(basedir, date, drive)

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
test_list = os.path.join(dataset.data_path, "test.txt")
test_txt = open(test_list, 'w')
plane_str = "# Plane\n" + "Width 4\n" + "Height 1\n" + "0 -1 0 1.65\n"
for i in tqdm(range(len(image_list))):
    # test.txt
    test_txt.write("%06d\n" % i)
    # plane file
    with open(os.path.join(plane_dir, "%06d.txt" % i), 'w') as f:
        f.write(plane_str)
    # calib file
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
test_txt.close()

# BEV
# bev_dir = os.path.join(dataset.data_path, "bev")
# os.system("rm -rf {}".format(bev_dir))
# os.mkdir(bev_dir)
#
# for i in tqdm(range(len(image_list))):
#     velo_data = dataset.get_velo(i)
#     fig = plt.figure(figsize=(5, 10))
#     ax = fig.add_subplot(111)
#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-20, 50)
#     ax.set_aspect('equal')
#     velo_range = range(0, velo_data.shape[0], 5)
#     ax.scatter(velo_data[velo_range, 1],
#                velo_data[velo_range, 0],
#                c=velo_data[velo_range, 3],
#                s=0.1,
#                cmap='rainbow')
#     ax.set_title('Velodyne scan (Bird\'s Eye View)')
#     plt.savefig(os.path.join(bev_dir, "%06d.png" % i), dpi=150, bbox_inches='tight')
#     plt.close('all')

# projecting point cloud to image
# def prepare_velo_points(pts3d_raw):
#     '''Replaces the reflectance value by 1, and tranposes the array, so
#        points can be directly multiplied by the camera projection matrix'''
#
#     pts3d = pts3d_raw
#     # Reflectance > 0
#     # pts3d = pts3d[pts3d[:, 3] > 0, :]
#     pts3d[:, 3] = 1
#     return pts3d.transpose()
#
#
# def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
#     '''Project 3D points into 2D image. Expects pts3d as a 4xN
#        numpy array. Returns the 2D projection of the points that
#        are in front of the camera only an the corresponding 3D points.'''
#
#     # 3D points in camera reference frame.
#     pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
#
#     # Before projecting, keep only points with z>0
#     # (points that are in fronto of the camera).
#     idx = (pts3d_cam[2, :] >= 0)
#     pts2d_cam = Prect.dot(pts3d_cam[:, idx])
#
#     return idx, pts2d_cam / pts2d_cam[2, :]
#
#
# T_cam_velo = dataset.calib.T_cam0_velo_unrect
# R_rect_00 = dataset.calib.R_rect_00
# P_rect_02 = dataset.calib.P_rect_20
#
# projection_dir = os.path.join(dataset.data_path, "projection")
# os.system("rm -rf {}".format(projection_dir))
# projection_image_dir = os.path.join(projection_dir, "image_2")
# list(map(os.mkdir, [projection_dir, projection_image_dir]))
# os.system("cp -r {} {}".format(calib_dir, os.path.join(projection_dir, "calib")))
#
# for i in tqdm(range(len(image_list))):
#     cam_data = dataset.get_cam2(i)
#     velo_data = dataset.get_velo(i)
#     velo_range = range(0, velo_data.shape[0], 8)
#     velo_data = velo_data[velo_range, :]
#     reflectance = velo_data[:, 3].copy()
#     velo_prepared = prepare_velo_points(velo_data)
#     idx, b = project_velo_points_in_img(velo_prepared, T_cam_velo, R_rect_00, P_rect_02)
#     width, height, dpi = 1242, 375, 96
#     fig = plt.figure()
#     fig.set_size_inches([width / dpi, height / dpi])
#     ax = plt.Axes(fig, [0, 0, 1, 1])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.set_xlim(0, width)
#     ax.set_ylim(height, 0)
#     ax.set_aspect('equal')
#     ax.imshow(cam_data)
#     ax.scatter(b[0, :], b[1, :], s=0.8, c=reflectance[idx], cmap='gist_rainbow')
#     fig.savefig(os.path.join(projection_image_dir, "%06d.png" % i), dpi=dpi)
#     plt.close('all')
#
# # move data to right place
list(map(lambda x: os.system("cp -r {} {} && rm -rf {}".format(x, "/mnt/Inside/Kitti/object", x)),
         [testing_dir, test_list]))
