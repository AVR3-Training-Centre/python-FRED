import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import numpy as np
import open3d as o3d
from utils.lidar import PointCloud
from utils.camera import ImageData
from natsort import natsorted
from utils.utils import get_corr_files
from sklearnex import patch_sklearn
patch_sklearn()
from tqdm import tqdm
import hazard_detection.map_based_utils as mb_utils

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

cmap = plt.get_cmap("jet")
verbose = False

# location        = 'Cambogan'
# sequence        = '20250811_113017'
# # location        = 'Holmview'
# # sequence        = '20250820_130327'
# # location        = 'DairyCreek'
# # sequence        = '20250811_103318'
# # location        = 'Pullenvale'
# # sequence        = '20250916_124105'
condition       = 'flooded'
camera_pos      = 'front'
# root_directory  = f"D:/Datasets/FRED/{condition}/KITTI-style" #f"C:/Users/conno/Documents/data/FRED/{condition}/KITTI-style" f"U:/Research/Projects/KVFPRA9190/FRED/{condition}/KITTI-style" 
# # root_directory  = f"X:/KVFPRA9190/FRED/{condition}/KITTI-style" 

# ref_location        = 'Cambogan'
# ref_sequence        = '20250812_122339'
# # ref_location        = 'Holmview'
# # ref_sequence        = '20250812_120100'
# # ref_location        = 'Dairy-Creek'
# # ref_sequence        = '20250812_122954'
# # ref_location        = 'Pullenvale'
# # ref_sequence        = '20250812_134316'
ref_condition       = 'dry'
# ref_root_directory  = f"D:/Datasets/FRED/{ref_condition}/KITTI-style" #f"C:/Users/conno/Documents/data/FRED/{ref_condition}/KITTI-style" f"U:/Research/Projects/KVFPRA9190/FRED/{ref_condition}/KITTI-style"
# # ref_root_directory  = f"X:/KVFPRA9190/FRED/{ref_condition}/KITTI-style"

locations = ['Cambogan', 'Holmview', 'DairyCreek', 'Pullenvale']
sequences = ['20250811_113017', '20250820_130327', '20250811_103318', '20250916_124105']
ref_locations = ['Cambogan', 'Holmview', 'Dairy-Creek', 'Pullenvale']
ref_sequences = ['20250812_122339', '20250812_120100', '20250812_122954', '20250812_134316']

for idx, location in enumerate(locations):

    print(f"Creating labels for {location}")

    root_directory  = f"D:/Datasets/FRED/{condition}/KITTI-style"
    ref_root_directory  = f"D:/Datasets/FRED/{ref_condition}/KITTI-style"
    ref_location = ref_locations[idx]
    sequence = sequences[idx]
    ref_sequence = ref_sequences[idx]


    save_dir = f"{root_directory}/{location}_{sequence}/"

    os.makedirs(f"{save_dir}/semantic_labels_dry/", exist_ok=True)
    os.makedirs(f"{save_dir}/semantic_labels_flooded/", exist_ok=True)

    image_dir       = f"{root_directory}/{location}_{sequence}/{camera_pos}-imgs/"
    label_dir       = f"{root_directory}/{location}_{sequence}/{camera_pos}-labels/"
    lidar_dir       = f"{root_directory}/{location}_{sequence}/ouster/"
    utm_dir         = f"{root_directory}/{location}_{sequence}/utm/"

    ref_lidar_dir   = f"{ref_root_directory}/{ref_location}_{ref_sequence}/ouster/"
    ref_utm_dir     = f"{ref_root_directory}/{ref_location}_{ref_sequence}/utm/"

    img_calib_file      = f"./camera_calib.txt"
    lidar_calib_file    = f"./calib.txt"

    timestamps = [fn.split('.bin')[0]
                for fn in natsorted(os.listdir(lidar_dir))
                if os.path.isfile(lidar_dir + fn)]

    # ─────────────────────────────────────────────────────────────────────────────
    # PRE-LOAD all reference UTMs once — was being reloaded every frame
    # ─────────────────────────────────────────────────────────────────────────────
    ref_utm_fnames = np.array([fn for fn in natsorted(os.listdir(ref_utm_dir))
                                if os.path.isfile(ref_utm_dir + fn)])
    ref_utms = np.array([np.loadtxt(ref_utm_dir + fn) for fn in ref_utm_fnames])

    # Cache for ref pointcloud data keyed by ref filename — avoids re-loading the
    # same ref scan when consecutive query frames match the same reference frame.
    _ref_cache = {}

    # ─────────────────────────────────────────────────────────────────────────────
    # Main per-frame function
    # ─────────────────────────────────────────────────────────────────────────────

    BEV_X_RANGE   = (-30, 50)
    BEV_Y_RANGE   = (-30, 30)
    BEV_RESOLUTION = 0.15

    max_utm_dist = 3.0   # metres — tune to your trajectory spacing / GPS noise;
                        # too loose and ICP may register to a non-corresponding
                        # location, too tight and you'll drop valid frames
    skipped_utm_too_far = 0

    for timestamp in tqdm(timestamps): # [100::]
        if verbose:
            print(f"Attempting pointcloud: {timestamp}.bin")

        if os.path.isfile(f"{save_dir}/semantic_labels_flooded/{timestamp}.label"):
            # print(f"{location}_{sequence}/{timestamp}.bin has already been processed.")
            # print(f"Skipping for now.")
            continue

        # ── UTM proximity filter — check BEFORE loading point clouds ─────────
        utm_filename, image_filename = get_corr_files(timestamp, [utm_dir, image_dir])
        qry_utm = np.loadtxt(utm_filename)
        f_temp         = image_filename.split("/")[-1]
        label_filename = f"{label_dir}/{f_temp}"
        image          = ImageData(image_filename, img_calib_file, label_filename)
        gt_mask        = np.zeros_like(image.image[:,:,0])
        pred_mask      = np.zeros_like(image.image[:,:,0])
        pointcloud     = PointCloud(f"{lidar_dir}/{timestamp}.bin",
                                    lidar_calib_file)


        # ── Load revisit (query) scan ────────────────────────────────────────
        rev_pointcloud = PointCloud(f"{lidar_dir}/{timestamp}.bin", lidar_calib_file)
        rev_xyz        = rev_pointcloud.points[:, :3]
        rev_intensity  = rev_pointcloud.points[:, 3]
        rev_ground     = rev_pointcloud.ground_semantic

        qry_label_mask, qry_label_points = mb_utils.get_gt(rev_pointcloud, None, image, cmap, full=True)
        qry_label_labels = qry_label_points[:,4].astype(np.uint32)
        qry_label_labels.tofile(f"{save_dir}/semantic_labels_flooded/{timestamp}.label")

        pc_dists = np.linalg.norm(ref_utms - qry_utm, axis=1)
        closest_dist = pc_dists[np.argmin(pc_dists)]
        if closest_dist > max_utm_dist:
            skipped_utm_too_far += 1
            if verbose:
                print(f"  Skipping {timestamp}: closest ref UTM {closest_dist:.2f}m away (> {max_utm_dist}m)")
            continue

        valid = ~np.all(rev_xyz == 0, axis=1)
        rev_xyz, rev_intensity, rev_ground = rev_xyz[valid], rev_intensity[valid], rev_ground[valid]

        ranges = np.linalg.norm(rev_xyz, axis=1)
        valid  = (ranges < 80) & (rev_xyz[:, 2] < 1)
        rev_xyz, rev_intensity, rev_ground = rev_xyz[valid], rev_intensity[valid], rev_ground[valid]

        # ── Find Closest Ref Frame ────────────────────────────────────────────────

        ref_xyz, ref_intensity, ref_ground, _ref_cache, ref_utm, ref_pc_filename = mb_utils.get_closest_ref_frame(qry_utm, ref_utms, 
                                                                                                    ref_utm_fnames, _ref_cache, 
                                                                                                    ref_lidar_dir, lidar_calib_file, verbose)

        ref_pointcloud = PointCloud(ref_pc_filename, lidar_calib_file)

        # ── ICP Registration ────────────────────────────────────────────────

        icp_voxel_size = 0.3
        rev_pcd, ref_pcd, rev_xyz_aligned, rev_down, result_icp = mb_utils.icp_registration(rev_xyz, ref_xyz, qry_utm, ref_utm, icp_voxel_size, verbose)

        T_inv = np.linalg.inv(result_icp.transformation)

        # ── Find Ground Truth Points ────────────────────────────────────────
        forward_angle_tol = 3
        max_lidar_range = 40
        has_gt = True

        gt_mask, gt_points = mb_utils.get_gt(ref_pointcloud, T_inv, image, cmap, full=True)
        gt_points_labels = gt_points[:,4].astype(np.uint32)
        gt_points_labels.tofile(f"{save_dir}/semantic_labels_dry/{timestamp}.label")

        # if len(gt_points) == 0 or gt_mask.max() == 0:
        #     has_gt = False
        # else:
        #     gt_dist, closest_gt_point, _ = mb_utils.get_closest_dist(gt_points, forward_angle_tol)

        #     if gt_dist > max_lidar_range:
        #         has_gt = False

        # gt_world = mb_utils.gt_points_to_world(gt_points, result_icp.transformation)
        
        # gt_bev_mask = mb_utils.bev_points_to_blob_mask(
        #     gt_world, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION,
        #     point_radius=4, close_kernel_size=9)

        