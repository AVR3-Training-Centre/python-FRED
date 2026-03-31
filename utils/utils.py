import numpy as np
import os
import cv2
import open3d as o3d
from natsort import natsorted

beam_altitudes = np.deg2rad(np.array([
    20.97, 18.51, 15.97, 13.37, 12.04, 10.70, 9.35, 8.70,
    7.99, 7.34, 6.62, 5.96, 5.23, 4.55, 3.84, 3.51,
    3.17, 2.82, 2.46, 2.11, 1.76, 1.43, 1.05, 0.70,
    0.36, 0.03, -0.34, -0.70, -1.04, -1.37, -1.76, -2.10,
    -2.43, -2.77, -3.15, -3.48, -3.84, -4.18, -4.54, -4.88,
    -5.24, -5.55, -5.93, -6.29, -6.63, -6.94, -7.32, -7.67,
    -8.00, -8.33, -9.04, -9.71, -10.42, -11.06, -11.77, -12.41,
    -13.12, -13.74, -15.06, -16.36, -17.64, -18.91, -20.16, -21.38
]))

def read_calib_file(path):
    """
    Read KITTI-style calibration file lines like:
    parameter: <value>
    parameter: <value>
    ...
    Returns a dict mapping keys to numpy arrays.
    """
    d = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line or line.startswith('#'):
                continue
            key, vals = line.split(':', 1)
            nums = [float(x) for x in vals.strip().split()]
            d[key.strip()] = np.array(nums)
    return d

def get_all_corr_files(image_timestamps, dirs, tol=200000, adjust=None):

    single = not isinstance(image_timestamps, (list, np.ndarray))
    if single:
        image_timestamps = [image_timestamps]

    image_timestamps = np.array(image_timestamps, dtype=np.int64)

    results_per_dir = []
    indices_per_dir = []
    for dir in dirs:
        if adjust is None:
            filenames = np.array([f for f in natsorted(os.listdir(dir)) if os.path.isfile(dir + f)])
        else:
            filenames = np.array([f for f in natsorted(os.listdir(dir)) if os.path.isfile(dir + f)])[:adjust]

        timestamps = np.array([int(f.split('.')[0]) for f in filenames], dtype=np.int64)

        diffs = np.abs(image_timestamps[:, None] - timestamps[None, :])
        closest_indices = np.argmin(diffs, axis=1)
        closest_diffs = diffs[np.arange(len(image_timestamps)), closest_indices]

        violations = closest_diffs > tol
        if violations.any():
            bad_ts = image_timestamps[violations]
            raise Exception(f"No timestamp in {dir} within tolerance for timestamps: {bad_ts}")

        results_per_dir.append(filenames[closest_indices])
        indices_per_dir.append(closest_indices)

    if len(dirs) == 1:
        results = [f"{dirs[0]}/{f}" for f in results_per_dir[0]]
        indices = indices_per_dir[0].tolist()
    else:
        results = [
            tuple(f"{dirs[i]}/{results_per_dir[i][j]}" for i in range(len(dirs)))
            for j in range(len(image_timestamps))
        ]
        indices = [
            tuple(int(indices_per_dir[i][j]) for i in range(len(dirs)))
            for j in range(len(image_timestamps))
        ]

    if single:
        return results[0], indices[0]
    return results, indices

def get_corr_files(image_timestamp, dirs, tol=200000):
    output_filenames = []
    for dir in dirs:
        ############ Find closest point cloud and UTM position data ########################
        filenames = np.array([filename for filename in os.listdir(dir) if os.path.isfile(dir+filename)])
        timestamps = np.array([int(filename.split('.')[0]) for filename in os.listdir(dir) if os.path.isfile(dir+filename)])

        closest_lidar = np.argmin(abs(timestamps-int(image_timestamp)))
        timestamp_diff = abs(int(image_timestamp)-timestamps[closest_lidar])

        timestamp_tolerance = tol # in microseconds (0.2 seconds)

        if timestamp_diff > timestamp_tolerance:
            raise Exception(f"No timestamp in {dir} in close enough proximity to {image_timestamp}")
        else:
            output_filenames.append(f"{dir}/{filenames[closest_lidar]}")

    if len(dirs) == 1:
        return output_filenames[0]
    else:
        return tuple(output_filenames)
    
NUM_COLS = 1024

def compute_column_index(points_xyz):
    az = np.arctan2(points_xyz[:, 1], points_xyz[:, 0])
    az = np.mod(az, 2 * np.pi)
    col = np.round(az / (2 * np.pi) * NUM_COLS).astype(int)
    return col % NUM_COLS

# beam_altitudes = np.deg2rad(np.array(beam_altitude_angles))

def compute_ring_ids(points_xyz):
    xy_norm = np.linalg.norm(points_xyz[:, :2], axis=1)
    elev = np.arctan2(points_xyz[:, 2], xy_norm)

    return np.argmin(
        np.abs(elev[:, None] - beam_altitudes[None, :]),
        axis=1
    )

def compute_ring_col_from_index(num_points, num_cols=1024):
    idx = np.arange(num_points)
    ring_ids = idx // num_cols
    col_ids  = idx % num_cols
    return ring_ids, col_ids

def is_valid_point(points_xyz):
    return np.linalg.norm(points_xyz, axis=1) > 1e-6

def fill_ring_known_cols_with_intensity(points_xyzi, ring_ids, col_ids):
    """
    points_xyzi: (N,4) -> x,y,z,intensity
    """
    filled_points = []
    interp_flags = []

    unique_ring_ids = np.array([i for i in range(0,64)])

    for ring in unique_ring_ids:
        ring_mask = ring_ids == ring
        ring_pts = points_xyzi[ring_mask]
        ring_cols = col_ids[ring_mask]

        # storage per column
        ring_grid = [None] * NUM_COLS

        # place original points
        for p, c in zip(ring_pts, ring_cols):
            ring_grid[c] = p  # keep intensity

        elev = beam_altitudes[ring]
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        for c in range(NUM_COLS):
            if ring_grid[c] is not None:
                filled_points.append(ring_grid[c])
                interp_flags.append(False)
            else:
                # find neighbors cyclically
                left = (c - 1) % NUM_COLS
                right = (c + 1) % NUM_COLS

                while ring_grid[left] is None:
                    left = (left - 1) % NUM_COLS
                while ring_grid[right] is None:
                    right = (right + 1) % NUM_COLS

                p0, p1 = ring_grid[left], ring_grid[right]

                r0 = np.linalg.norm(p0[:3])
                r1 = np.linalg.norm(p1[:3])

                d = (c - left) % NUM_COLS
                span = (right - left) % NUM_COLS
                t = d / span if span > 0 else 0.0
                r = (1 - t) * r0 + t * r1

                az = 2 * np.pi * c / NUM_COLS

                x = r * cos_e * np.cos(az)
                y = r * cos_e * np.sin(az)
                z = r * sin_e

                filled_points.append([x, y, z, 255])
                interp_flags.append(True)

    return np.array(filled_points), np.array(interp_flags)

def fill_ring_known_cols_with_intensity_and_plane(
    points_xyzi,
    ring_ids,
    col_ids,
    plane,
    max_range=200.0
):
    """
    points_xyzi: (N,4) -> x,y,z,intensity
    """
    a, b, c, d = plane

    filled_points = []
    interp_flags = []

    unique_ring_ids = np.array([i for i in range(0,64)])

    for ring in unique_ring_ids:
        # print(f"ring {ring}")
        ring_mask = ring_ids == ring
        ring_pts = points_xyzi[ring_mask]
        ring_cols = col_ids[ring_mask]

        ring_grid = [None] * NUM_COLS
        # print(f"ring grid shape: {np.array(ring_grid).shape}")
        

        # place original points
        for p, col in zip(ring_pts, ring_cols):
            ring_grid[col] = p

        elev = beam_altitudes[ring]
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        for c_idx in range(NUM_COLS):

            # print(f"c_idx: {c_idx}")

            if ring_grid[c_idx] is not None:
                filled_points.append(ring_grid[c_idx])
                interp_flags.append(False)
                continue

            # find neighbors cyclically
            # print(f"Find neighbours")
            # left = (c_idx - 1) % NUM_COLS
            # right = (c_idx + 1) % NUM_COLS

            # while ring_grid[left] is None:
            #     left = (left - 1) % NUM_COLS
            # while ring_grid[right] is None:
            #     right = (right + 1) % NUM_COLS

            # print(f"p0, p1")

            # p0, p1 = ring_grid[left], ring_grid[right]

            # r0 = np.linalg.norm(p0[:3])
            # r1 = np.linalg.norm(p1[:3])

            # print(f"dcol, span")

            # dcol = (c_idx - left) % NUM_COLS
            # span = (right - left) % NUM_COLS
            # t_interp = dcol / span if span > 0 else 0.0

            # r_guess = (1 - t_interp) * r0 + t_interp * r1

            # print(f"az")

            az = 2 * np.pi * c_idx / NUM_COLS

            # Ray direction
            dx = cos_e * np.cos(az)
            dy = cos_e * np.sin(az)
            dz = sin_e

            denom = a * dx + b * dy + c * dz
            if abs(denom) < 1e-6:
                # print(f"point in rang {ring} is parallel")
                continue  # ray parallel to plane

            t_plane = -d / denom
            # if t_plane <= 0 or t_plane > max_range:
            #     # print(f"point in rang {ring} is {t_plane}m away")
            #     continue

            # print(f"Fill point")

            x = t_plane * dx
            y = t_plane * dy
            z = t_plane * dz

            filled_points.append([x, y, z, 255])
            interp_flags.append(True)

    return np.asarray(filled_points), np.asarray(interp_flags)


def complete_cloud(points_xyzi, plane):
    # ring_ids = compute_ring_ids(points_xyzi[:, :3])
    # col_ids = compute_column_index(points_xyzi[:, :3])
    ring_ids, col_ids = compute_ring_col_from_index(
        len(points_xyzi), NUM_COLS
    )
    valid_mask = is_valid_point(points_xyzi[:, :3])

    # return fill_ring_known_cols_with_intensity_and_plane(points_xyzi, ring_ids, col_ids, plane)
    return fill_ring_known_cols_with_intensity_and_heightfield(points_xyzi, ring_ids, col_ids, valid_mask, plane)


def assign_semantic_labels(
    points_xyz,
    uv,
    valid,
    semantic_image,
    interp_flags=None,
    unknown_label=-1
):
    """
    semantic_image: (H, W) or (H, W, 1) integer labels
    """
    H, W = semantic_image.shape[:2]
    N = points_xyz.shape[0]

    labels = np.full(N, unknown_label, dtype=np.uint8)

    # round to nearest pixel
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)

    inside = (
        valid &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    labels[inside] = semantic_image[v[inside], u[inside]]

    # Optional: mark interpolated points as unknown
    if interp_flags is not None:
        labels[interp_flags] = unknown_label

    return labels

def fit_height_field_linear(ground_points):
    """
    Fit z = ax + by + c to ground points
    """
    X = ground_points[:, 0]
    Y = ground_points[:, 1]
    Z = ground_points[:, 2]

    A = np.column_stack((X, Y, np.ones_like(X)))
    coef, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    a, b, c = coef
    return a, b, c

def fill_ring_known_cols_with_intensity_and_heightfield(
    points_xyzi,
    ring_ids,
    col_ids,
    valid_mask,
    height_field,   # (a, b, c)
    max_range=200.0
):
    """
    points_xyzi: (N,4) -> x,y,z,intensity
    """
    a, b, c = height_field

    filled_points = []
    interp_flags = []

    for ring in range(64):
        ring_mask = ring_ids == ring

        ring_pts   = points_xyzi[ring_mask]
        ring_cols  = col_ids[ring_mask]
        ring_valid = valid_mask[ring_mask]

        ring_grid = [None] * NUM_COLS

        # Place ONLY valid original points
        for p, col, v in zip(ring_pts, ring_cols, ring_valid):
            if v:
                ring_grid[col] = p

        elev = beam_altitudes[ring]
        cos_e, sin_e = np.cos(elev), np.sin(elev)

        for c_idx in range(NUM_COLS):

            if ring_grid[c_idx] is not None:
                filled_points.append(ring_grid[c_idx])
                interp_flags.append(False)
                continue

            # Missing return → ray cast
            az = 2 * np.pi * c_idx / NUM_COLS

            dx = cos_e * np.cos(az)
            dy = cos_e * np.sin(az)
            dz = sin_e

            denom = dz - a * dx - b * dy
            if abs(denom) < 1e-6:
                continue

            t = c / denom
            if t <= 0 or t > max_range:
                continue

            x = t * dx
            y = t * dy
            z = t * dz

            filled_points.append([x, y, z, 255])
            interp_flags.append(True)

    return np.asarray(filled_points), np.asarray(interp_flags)
    # a, b, c = height_field

    # filled_points = []
    # interp_flags = []

    # unique_ring_ids = np.arange(64)

    # for ring in unique_ring_ids:
    #     ring_mask = ring_ids == ring
    #     ring_pts = points_xyzi[ring_mask]
    #     ring_cols = col_ids[ring_mask]

    #     ring_grid = [None] * NUM_COLS

    #     # place original points
    #     for p, col in zip(ring_pts, ring_cols):
    #         ring_grid[col] = p

    #     elev = beam_altitudes[ring]
    #     cos_e, sin_e = np.cos(elev), np.sin(elev)

    #     for c_idx in range(NUM_COLS):

    #         if ring_grid[c_idx] is not None:
    #             filled_points.append(ring_grid[c_idx])
    #             interp_flags.append(False)
    #             continue

    #         az = 2 * np.pi * c_idx / NUM_COLS

    #         # Unit ray direction
    #         dx = cos_e * np.cos(az)
    #         dy = cos_e * np.sin(az)
    #         dz = sin_e

    #         denom = dz - a * dx - b * dy
    #         if abs(denom) < 1e-6:
    #             continue  # ray parallel to height field

    #         t = c / denom
    #         if t <= 0 or t > max_range:
    #             continue

    #         x = t * dx
    #         y = t * dy
    #         z = t * dz

    #         filled_points.append([x, y, z, 255])
    #         interp_flags.append(True)

    # return np.asarray(filled_points), np.asarray(interp_flags)

def create_height_field_mesh(plane, xlim, ylim, resolution=1.0):
    a, b, c = plane
    xs = np.arange(xlim[0], xlim[1], resolution)
    ys = np.arange(ylim[0], ylim[1], resolution)

    xx, yy = np.meshgrid(xs, ys)
    zz = a * xx + b * yy + c

    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)

    triangles = []
    w = len(xs)
    h = len(ys)

    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            triangles.append([i, i + 1, i + w])
            triangles.append([i + 1, i + w + 1, i + w])

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

    return mesh
