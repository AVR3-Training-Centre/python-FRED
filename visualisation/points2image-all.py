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

# Toggle the following boolean to False if not using HuggingFace App
hf_app = False

if hf_app:
    from huggingface_hub import snapshot_download

cmap = plt.get_cmap("jet")

# User parameters
location = 'Cambogan'
sequence = '20250811_113017'
condition = 'flooded'
camera_pos = 'front'
root_directory = f"D:/Datasets/FRED/{condition}/KITTI-style"

if (not os.path.exists(root_directory)) and (hf_app):
    snapshot_download(
        repo_id="CMalone-Jupiter/FRED",
        repo_type="dataset",
        local_dir="/data/FRED",
        allow_patterns=f"{condition}/KITTI-style/{location}_{sequence}/**",
        token=os.environ.get("HF_TOKEN")
    )

############ Define filenames and directories ####################################

image_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
lidar_dir = f"{root_directory}/{location}_{sequence}/ouster/"
utm_dir = f"{root_directory}/{location}_{sequence}/utm/"

img_calib_file = f"./camera_calib.txt"
# lidar_calib_file = f"./calib.txt"
lidar_calib_file = f"./paper_calib.txt"

timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]


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


        point_cam, distances_cam, intensities_cam, all_points_cam, valid_cam = pointcloud.points_ouster_to_cam() #, beam_id, azimuth

        p_high = np.percentile(intensities_cam, 99)
        intensities_cam = np.clip(intensities_cam, 0, p_high) / p_high * 255

        img_vis, _, _, _ = image.project_points(all_points_cam, intensities_cam, cmap, valid_cam, colour_norm=255) #, beam_id, azimuth

        ax.imshow(img_vis[:, :, ::-1])
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