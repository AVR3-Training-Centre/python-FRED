import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append(os.path.abspath("."))   # one level up
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from utils.lidar import PointCloud
from utils.camera import ImageData
import utils.utils as utils
from natsort import natsorted

cmap = plt.get_cmap("jet")

# User parameters
location = 'Cambogan'
sequence = '20250811_113017'
# location = 'Holmview'
# sequence = '20250820_130327'
# location = 'Mount-Cotton'
# sequence = '20241217_113410'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"D:/Datasets/FRED/{condition}/KITTI-style"
# 01000000

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
lidar_dir = f"{root_directory}/{location}_{sequence}/ouster/"
utm_dir = f"{root_directory}/{location}_{sequence}/utm/"

img_calib_file = f"./camera_calib.txt"
lidar_calib_file = f"./calib.txt"

timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]

# timestamps.sort()

fig, ax = plt.subplots(figsize=(12.8, 8))
idx = [0]  # mutable index

def show_image(i):
    ax.clear()
    if i >= len(timestamps):
        plt.close(fig)
        return
    image_timestamp = timestamps[i]
    try:
        image_filename = f"{image_dir}/{image_timestamp}.png"
        lidar_filename, utm_filename = utils.get_corr_files(image_timestamp, [lidar_dir, utm_dir])

        image = ImageData(image_filename, img_calib_file)
        pointcloud = PointCloud(lidar_filename, lidar_calib_file)

        lateral_filter = np.logical_and(-4 < pointcloud.points[:,1], pointcloud.points[:,1] < 1.5)
        ground_filter = pointcloud.ground_semantic == 0
        inlier_filter = pointcloud.ground_inlier == 1
        points_filter = np.logical_and(np.logical_and(lateral_filter, ground_filter), inlier_filter)
        filtered_points = pointcloud.points[points_filter]
        pointcloud.points = filtered_points
        max_lookahead = pointcloud.points[:,0].max()
        far_points = pointcloud.points[pointcloud.points[:,0]==max_lookahead,:]

        if far_points.shape[0] > 1:
            # pointcloud.points = far_points[abs(far_points[:,1]) == abs(far_points[:,1]).min(),:]
            far_point = far_points[abs(far_points[:,1]) == abs(far_points[:,1]).min(),:]
        else:
            # pointcloud.points = far_points
            far_point = far_points

        points_cam, distances_cam, intensities_cam, beam_id, azimuth = pointcloud.points_ouster_to_cam()
        far_point_cam, far_point_distnace, far_point_intensity = pointcloud.select_points_ouster_to_cam(far_point)
        img_vis = image.project_points(np.vstack((points_cam, far_point_cam)), np.append(intensities_cam, 128), beam_id, azimuth, cmap)
        far_pixel = image.get_image_coords(far_point_cam)

        if far_pixel is not None and len(far_pixel) > 0:
            u, v = far_pixel[0]  # pixel coordinates

        h, w = img_vis.shape[:2]
        bottom_center = (w // 2, h)

        ax.imshow(img_vis[:, :, ::-1])
        ax.plot(
            [bottom_center[0], u],
            [bottom_center[1], v],
            color="lime",
            linewidth=2
        )
        ax.text(
            u,
            v - 10,
            f"{far_point[0,0]:.2f}",
            color="lime",
            fontsize=12,
            ha="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
        )
        ax.set_title(f"{image_timestamp}.png")
        ax.axis("off")
        fig.canvas.draw()
    except Exception as e:
        print(f"Could not project pointcloud onto {image_timestamp}.png: {e}")
        idx[0] += 1
        show_image(idx[0])  # skip bad one

def on_key(event):
    if event.key in [' ', 'right']:  # space or right arrow
        idx[0] += 1
        show_image(idx[0])
    elif event.key in [' ', 'left']:  # space or right arrow
        if idx[0] > 0:
            idx[0] -= 1
            show_image(idx[0])
    elif event.key in ['q', 'escape']:  # q or Esc → quit
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
show_image(idx[0])
plt.show()