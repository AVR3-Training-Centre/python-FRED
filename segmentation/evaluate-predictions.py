import os
import argparse
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
import json
import yaml  # pip install pyyaml
from tqdm import tqdm

# Toggle the following boolean to False if not using HuggingFace App
hf_app = False

if hf_app:
    from huggingface_hub import snapshot_download

cmap = plt.get_cmap("jet")

# ---------------- Argument Parsing ---------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Path to config file (YAML or JSON). If provided, overrides other args.")

parser.add_argument("--location", type=str, default="Cambogan", help="Location name (e.g., Cambogan)")
parser.add_argument("--sequence", type=str, default="20250811_113017", help="Sequence ID (e.g., 20250811_113017)")
parser.add_argument("--condition", type=str, default="flooded", help="Condition (e.g., flooded)")
parser.add_argument("--camera_pos", type=str, default="front", help="Camera position (e.g., front)")
parser.add_argument("--root", type=str, default="/data/FRED/", help="Root dataset directory (e.g., ../Datasets/FRED/)")
parser.add_argument("--masks", type=str, required=True, help="Where predicted masks are saved")
parser.add_argument("--img_calib_file", type=str, default="./camera_calib.txt", help="Path to camera calibration file (e.g., ./camera_calib.txt)")
parser.add_argument('--vis', action='store_true', help="Store visual comparisons of the predictions and labels")
parser.add_argument("--output", type=str, default=None, help="Where to save visual comparisons")

args = parser.parse_args()

# ---------------- Config Loading ---------------- #
if args.config:
    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    elif args.config.endswith(".json"):
        with open(args.config, "r") as f:
            cfg = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    location = cfg["location"]
    sequence = cfg["sequence"]
    condition = cfg["condition"]
    camera_pos = cfg["camera_pos"]
    root = cfg["root"]
    root_directory = f"{root}/{condition}/KITTI-style"
    img_calib_file = cfg["img_calib_file"]

else:
    # Fallback: require all CLI args
    required_args = ["location", "sequence", "condition", "camera_pos", "root", "img_calib_file"]
    missing = [arg for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing arguments: {', '.join(missing)} (or provide --config)")

    location = args.location
    sequence = args.sequence
    condition = args.condition
    camera_pos = args.camera_pos
    root_directory = f"{args.root}/{args.condition}/KITTI-style"
    img_calib_file = args.img_calib_file

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
label_dir = f"{root_directory}/{location}_{sequence}/{camera_pos}-labels/"
img_calib_file = f"./camera_calib.txt"
timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(image_dir)) if os.path.isfile(image_dir+filename)]
mask_filenames = [filename for filename in natsorted(os.listdir(args.masks)) if os.path.isfile(args.masks+filename)]


fig, ax = plt.subplots(figsize=(12.8, 8))
idx = 0  # mutable index

def load_image_data(i):
    image_timestamp = timestamps[i]
    try:
        image_filename = f"{image_dir}/{image_timestamp}.png"
        label_filename = f"{label_dir}/{image_timestamp}.png"
        image = ImageData(image_filename, img_calib_file, label_filename)
        water_label = image.label_img == 1

        return water_label.astype(int)

    except Exception as e:
        print(f"Could not show label for {image_timestamp}.png: {e}")




def calculate_iou(label: np.ndarray, prediction: np.ndarray, verbose=False) -> float:
    """
    Calculate Intersection over Union (IOU) for two binary masks.
    
    Args:
        label:      Ground truth binary mask (values 0 or 1)
        prediction: Predicted binary mask (values 0 or 1)
    
    Returns:
        IOU score as a float in [0, 1]. Returns 0.0 if both masks are empty.
    """
    if label.shape != prediction.shape:
        # raise ValueError(f"Shape mismatch: label {label.shape} vs prediction {prediction.shape}")
        if verbose:
            print(f"reshaping predictions from {prediction.shape} to {label.shape} to match labels.")
        prediction = cv2.resize(prediction, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(label, prediction).sum()
    union = np.logical_or(label, prediction).sum()

    if union == 0:
        return 1.0  # or 1.0 if you want to treat two empty masks as a perfect match

    return float(intersection / union)

def save_mask_comparison(label: np.ndarray, prediction: np.ndarray, save_path: str, iou: float | None = None) -> None:
    """
    Save label and prediction masks side by side for visual comparison.

    Args:
        label:      Ground truth binary mask
        prediction: Predicted binary mask
        save_path:  Path to save the output image (e.g. 'comparison.png')
        iou:        Optional IOU score to display in the title
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(label, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Label')
    axes[0].axis('off')

    axes[1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    title = 'Mask Comparison'
    if iou is not None:
        title += f' | IOU: {iou:.4f}'
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

iou_scores = []

for i in tqdm(range(idx, len(timestamps))):

    img_label = load_image_data(i)
    pred_label = (cv2.cvtColor(cv2.imread(args.masks+mask_filenames[i]), cv2.COLOR_BGR2GRAY)/255).astype(int)

    iou = calculate_iou(img_label, pred_label)
    iou_scores.append(iou)
    if args.vis:
        os.makedirs(args.output, exist_ok=True)
        save_mask_comparison(img_label, pred_label, f"{args.output}/{timestamps[i]}.png", iou)


print(f"Mean IOU for water predictions: {np.mean(np.array(iou_scores))}")
