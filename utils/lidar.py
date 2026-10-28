import numpy as np
import os
from utils.utils import read_calib_file #, compute_os1_angles, elevation_to_beam_id

INLIER_BIT = 1 << 8
INSTANCE_SHIFT = 16
beam_altitude_angles_deg = np.array([
    20.97, 18.51, 15.97, 13.37, 12.04, 10.70, 9.35, 8.70,
    7.99, 7.34, 6.62, 5.96, 5.23, 4.55, 3.84, 3.51,
    3.17, 2.82, 2.46, 2.11, 1.76, 1.43, 1.05, 0.70,
    0.36, 0.03, -0.34, -0.70, -1.04, -1.37, -1.76, -2.10,
    -2.43, -2.77, -3.15, -3.48, -3.84, -4.18, -4.54, -4.88,
    -5.24, -5.55, -5.93, -6.29, -6.63, -6.94, -7.32, -7.67,
    -8.00, -8.33, -9.04, -9.71, -10.42, -11.06, -11.77, -12.41,
    -13.12, -13.74, -15.06, -16.36, -17.64, -18.91, -20.16, -21.38
])

class PointCloud:
    def __init__(self, file_path, calib_path):

        self.points = self.load_pointcloud(file_path)
        ground_labels = np.fromfile(f"{file_path.split('/ouster')[0]}/ouster_ground_labels/{file_path.split('/')[-1].split('.bin')[0]}.label", dtype=np.uint32)
        # points = self.load_pointcloud(file_path)
        # self.points = points[self.ground_labels==0,:]
        self.ground_semantic = ground_labels & 0xFF
        self.ground_inlier = (ground_labels & INLIER_BIT) != 0
        
        calibration = read_calib_file(calib_path)
        self.P2, self.R0, self.Tr4 = self.get_matrices(calibration)

        self.pixel_shift_by_row = [12, 12, 12, 12, 12, 12, 12, -4, 
                                   12, -4, 12, -4, 12, -4, 12, 4, 
                                   -4, -12, 12, 4, -4, -12, 12, 4, 
                                   -4, -12, 12, 4, -4, -12, 12, 4, 
                                   -4, -12, 12, 4, -4, -12, 12, 4, 
                                   -4, -12, 12, 4, -4, -12, 12, 4, 
                                   -4, -12, 4, -12, 4, -12, 4, -12, 
                                   4, -12, -12, -12, -12, -12, -12, -12]


    
    def load_pointcloud(self, file_path):
        """
        Load point cloud from various formats including .bin files
        Assumes each point has 4 values (x, y, z, reflectivity)
        
        Args:
            file_path (str): Path to point cloud file
            
        Returns:
            np.array: Nx4 array of 3D points with 
        """

        file_ext = os.path.splitext(file_path)[1].lower()
            
        if not file_ext == '.bin':
            raise ValueError(f"Given file is not binary format: {file_path}")
        
        points = np.fromfile(file_path, dtype=np.float32)

        if len(points) % 4 == 0:
            points_array = points.reshape(-1, 4)

        return points_array
    

    def get_matrices(self,calib):
        # P2 (3x4)
        P2 = calib['P2'].reshape(3,4)
        # R0_rect (3x3) or identity
        R0 = calib.get('R0_rect', np.array([1,0,0,0,1,0,0,0,1])).reshape(3,3)
        # Tr_ouster_to_cam (3x4)
        Tr = calib['Tr_ouster_to_cam'].reshape(3,4)
        # Make 4x4 homogeneous
        Tr4 = np.vstack((Tr, [0,0,0,1]))

        return P2, R0, Tr4

    def points_ouster_to_cam(self,):
        """
        pts_ouster: (N,3) numpy array in LiDAR frame.
        Tr4: 4x4 homogeneous transform from ouster -> camera.
        Returns pts_cam (N,3) in camera coordinates.
        """
        n = self.points.shape[0]

        pts_xyz = self.points[:, :3]

        pts_h = np.column_stack((pts_xyz, np.ones((n))))        # (N,4)
        pts_cam_h = (self.Tr4 @ pts_h.T).T[:,:3]                       # (N,4)

        valid = pts_cam_h[:,2] > 1e-6

        # print(pts_cam_h.shape)
        # print(valid.shape)

        distances = np.linalg.norm(pts_cam_h[valid,:], axis=1)     

        return pts_cam_h[valid,:], np.linalg.norm(pts_cam_h, axis=1), self.points[:,3], pts_cam_h, valid
    
    def select_points_ouster_to_cam(self, points):
        """
        pts_ouster: (N,3) numpy array in LiDAR frame.
        Tr4: 4x4 homogeneous transform from ouster -> camera.
        Returns pts_cam (N,3) in camera coordinates.
        """
        n = points.shape[0]
        pts_h = np.column_stack((points[:,:3], np.ones((n))))        # (N,4)
        pts_cam_h = (self.Tr4 @ pts_h.T).T[:,:3]                       # (N,4)

        valid = pts_cam_h[:,2] > 1e-6
        # valid = np.ones((pts_cam_h.shape[0]))

        distances = np.linalg.norm(pts_cam_h[valid,:], axis=1)        

        return pts_cam_h[valid,:], distances, points[valid,3]
    
    def destagger(self):

        pixel_shift_by_row = np.array(self.pixel_shift_by_row)

        H = 64
        W = 1024

        # pc_img = points.reshape(W, H, 4).transpose(1, 0, 2)
        # # shape: (64, 1024, 4)

        # rows = np.arange(H)[:, None]
        # cols = np.arange(W)[None, :]

        # pc_destaggered = pc_img[
        #     rows,
        #     (cols - pixel_shift_by_row[:, None]) % W,
        #     :
        # ]

        # return pc_destaggered.reshape(-1, 4)

        
        # --- reshape ---
        pc_img = self.points.reshape(W, H, 4).transpose(1, 0, 2)   # (64, 1024, 4)
        lbl_img = self.ground_semantic.reshape(W, H).T                      # (64, 1024)
        inlier_img = self.ground_inlier.reshape(W, H).T 

        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]

        # --- destagger (IDENTICAL indexing) ---
        pc_destaggered = pc_img[
            rows,
            (cols - pixel_shift_by_row[:, None]) % W,
            :
        ]

        lbl_destaggered = lbl_img[
            rows,
            (cols - pixel_shift_by_row[:, None]) % W
        ]

        inlier_destaggered = inlier_img[
            rows,
            (cols - pixel_shift_by_row[:, None]) % W
        ]

        return pc_destaggered.reshape(-1, 4), lbl_destaggered.reshape(-1), inlier_destaggered.reshape(-1)

