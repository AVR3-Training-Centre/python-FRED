<div align="center">

# FRED: Flooded Road Environments Dataset

**A multi-modal autonomous driving dataset for detecting water hazards on the road**

Connor Malone · Sébastien Demmel · Sébastien Glaser  
ARC Training Centre for Automated Vehicles in Rural and Remote Regions ([AVR3](https://avr3.org.au/)), Queensland University of Technology

[![arXiv](https://img.shields.io/badge/arXiv-2605.22018-b31b1b?style=flat)](https://arxiv.org/abs/2605.22018)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-FRED-FF9D00?style=flat)](https://huggingface.co/datasets/CMalone-Jupiter/FRED)
[![Space](https://img.shields.io/badge/🤗%20Space-python--FRED-FF9D00?style=flat)](https://huggingface.co/spaces/CMalone-Jupiter/python-FRED)
[![Collection: AVR3](https://img.shields.io/badge/collection-AVR3-orange?style=flat)](https://avr3.org.au/)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Dataset licence: CC BY-NC-SA 4.0](https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-red?style=flat)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Code licence: BSD-3](https://img.shields.io/badge/code-BSD--3-blue?style=flat)](./LICENSE)
[![stars](https://img.shields.io/github/stars/AVR3-Training-Centre/python-FRED?style=flat)](https://github.com/AVR3-Training-Centre/python-FRED/stargazers)

[📄 Paper](https://arxiv.org/abs/2605.22018) &nbsp;|&nbsp;
[🤗 Dataset](https://huggingface.co/datasets/CMalone-Jupiter/FRED) &nbsp;|&nbsp;
[🚀 Live demo (HF Space)](https://huggingface.co/spaces/CMalone-Jupiter/python-FRED) &nbsp;|&nbsp;
[🗂 QUT Research Data Finder](https://data.researchdatafinder.qut.edu.au/dataset/flooded-road-environments)

<img src="assets/Zoe2-FRED.svg" alt="The Zoe 2 data collection vehicle and a flooded road sample from FRED" width="80%">

</div>

## Updates

<!-- Add new entries at the top, newest first. -->
- **Preprint released** on [arXiv](https://arxiv.org/abs/2605.22018).
- **Dataset released** on [Hugging Face](https://huggingface.co/datasets/CMalone-Jupiter/FRED).
- **Development kit and interactive Space** available on [GitHub](https://github.com/AVR3-Training-Centre/python-FRED) and [Hugging Face](https://huggingface.co/spaces/CMalone-Jupiter/python-FRED).

## Overview

Water hazards such as flooded roads are a safety-critical perception problem for autonomous vehicles, yet very few public datasets contain labelled data from vehicles encountering them, and none combine a full autonomy sensor stack with them. FRED is, to our knowledge, the first multi-modal autonomous driving dataset that specifically targets water hazards on the road.

It was recorded with a Renault Zoe (the *Zoe 2*) modified for autonomous operation and contains front and rear camera images, 360° LiDAR point clouds, and GNSS-corrected IMU data from **five locations around Brisbane, Australia**, captured both **during/after flooding events** and, later, under **dry** conditions.

**Highlights**

- 📷 **Multi-modal**: 2.3 MP front and rear images, 64-beam 360° point clouds, and RTK GNSS/IMU, all time-aligned using centralised timestamps.
- 🌊 **Varied flood scenarios**: large static flood water, a flowing stream across the road, and puddling.
- 🏷️ **Semantic labels** for images (water hazard / road / other) and for point clouds, including labels for LiDAR void regions caused by water.
- 🗺️ **Position and dry re-visits**: centimetre-level trajectories and dry-condition runs of each location, supporting map-based detection, localisation and visual place recognition research.
- 📦 **Two formats**: a KITTI-style layout for easy integration with existing tools, and the native RTMaps format for direct replay.
- 📊 **Benchmarks** for image-based segmentation, LiDAR and sensor-fusion segmentation, and visual place recognition.

## Dataset at a Glance

| | |
|---|---|
| Samples | ≈ 5,340 (images, point clouds and IMU data) |
| Flooded-condition images | 2,726, of which 2,407 contain labelled water |
| Locations | 5 (Mount Cotton, Cambogan, Holmview, Pullenvale, Dairy Creek) |
| Conditions | Flooded, and dry re-visits for map building |
| Image labels | 3 classes: water hazard, road, other (flooded sequences, front camera) |
| Point cloud labels | Water-hazard labels transferred from images via co-located dry point clouds |
| KITTI-style sampling | ≈ 10 Hz |
| Positioning | WGS84 latitude/longitude and WGS84 / UTM zone 56S easting/northing |
| Privacy | Images anonymised with the [Understand-AI Anonymizer](https://github.com/fusionportable/Anonymizer) |

### Flooded sequences

| Location | Images | Water-labelled (%) | Flood type |
|---|---:|---:|---|
| Cambogan | 243 | 86.42 | Static flooding |
| Dairy Creek | 280 | 100.0 | Static flooding |
| Holmview | 420 | 71.19 | Static flooding |
| Mount Cotton | 1484 | 92.92 | Puddles |
| Pullenvale | 299 | 79.93 | Flowing water |

Labels cover water directly on the drivable surface. In the static-flooding sequences the vehicle approaches the water's edge, so the labelled water area grows through the sequence.

## Sensor Suite

| Sensor | Specification |
|---|---|
| FLIR Blackfly BFLY-U3-23S6C-C (front and rear) | 1920×1200 (2.3 MP), 49.1° HFOV × 31.9° VFOV, 30 Hz |
| Ouster OS1-64 LiDAR | 64 beams, 360° HFOV × ±22.5° VFOV, 0.7° vertical / 0.35° horizontal resolution, ≈ 0.65 M points/s, 10 Hz |
| iXblue ATLANS-C IMU | 200 Hz, 0.02° heading and 0.008° roll/pitch RTK accuracy |
| Geoflex RTK GNSS | 1–3 cm accuracy (RTK) |

Calibration files for the sensor transformations and camera projection ([`calib.txt`](calib.txt), [`camera_calib.txt`](camera_calib.txt)) are included in this repository. The vehicle reference point is centred over the rear axle, where the IMU is positioned. Images are provided rectified, so distortion coefficients can be set to zero.

## Getting the Data

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/CMalone-Jupiter/FRED). To download a single KITTI-style sequence:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CMalone-Jupiter/FRED",
    repo_type="dataset",
    local_dir="/data/FRED",
    allow_patterns="flooded/KITTI-style/Cambogan_20250811_113017/**",
)
```

To download everything:

```bash
huggingface-cli download CMalone-Jupiter/FRED --repo-type dataset --local-dir /data/FRED
```

The dataset is also listed in the [QUT Research Data Finder](https://data.researchdatafinder.qut.edu.au/dataset/flooded-road-environments).

## Installation

The development kit includes a `pixi.toml` for setting up a Python environment with the [pixi](https://pixi.sh) package manager:

```bash
git clone https://github.com/AVR3-Training-Centre/python-FRED.git
cd python-FRED
pixi install
pixi shell
```

> The `pixi.toml` currently targets `win-64`. On other platforms, add your platform to `platforms` in `pixi.toml`.

A `Dockerfile` is provided for the [Hugging Face Space](https://huggingface.co/spaces/CMalone-Jupiter/python-FRED), which lets you try the kit online with JupyterLab. It may also work for building a local Docker environment, but this has not been tested.

## Quick Start

Scripts under `segmentation/` accept a YAML config; examples for each flooded sequence are in [`configs/flooded/`](configs/flooded) (and dry sequences in [`configs/dry/`](configs/dry)). Edit `root` in the config to point at your copy of the dataset.

```bash
# Overlay semantic labels on images
python segmentation/show_labels-all.py --config configs/flooded/cambogan.yaml

# Evaluate predicted water masks against the labels (optionally save visual comparisons)
python segmentation/evaluate-predictions.py --config configs/flooded/cambogan.yaml --masks path/to/predicted_masks --vis
```

The visualisation scripts (e.g. [`visualisation/points2image-all.py`](visualisation/points2image-all.py)) are configured by editing the `location`, `sequence`, `condition` and `root_directory` variables at the top of the script. They also have an `hf_app` toggle: when `True`, missing sequences are downloaded automatically from Hugging Face. Set it to `False` when using a local copy of the dataset. Single-sample Jupyter notebooks (`*-single.ipynb`) and [`welcome.ipynb`](welcome.ipynb) provide interactive walkthroughs.

## Development Kit

| Tool | Location |
|---|---|
| Project point clouds onto images (colour by range, intensity or semantic label) | [`visualisation/`](visualisation) |
| Visualise semantic labels on images | [`segmentation/`](segmentation) |
| Evaluate semantic segmentation predictions (IoU) | [`segmentation/evaluate-predictions.py`](segmentation/evaluate-predictions.py) |
| Display corresponding images across sequences from the same location | [`localisation/`](localisation) |
| Plot and validate sequence trajectories | [`localisation/`](localisation) |
| Evaluate Visual Place Recognition (VPR) performance | [`localisation/`](localisation) |
| Create point cloud semantic labels from image annotations | [`lidar_postprocessing/`](lidar_postprocessing) |
| Create range images from point clouds | [`lidar_postprocessing/`](lidar_postprocessing) |
| Infill missing ground points in point clouds | [`lidar_postprocessing/fill_pointcloud_gaps.py`](lidar_postprocessing/fill_pointcloud_gaps.py) |
| Loading utilities for images, LiDAR and calibration | [`utils/`](utils) |
| Feature re-ranking tools | [`FoL/`](FoL) |

The kit will continue to be developed and updated.

## Dataset Structure

Each sequence is identified by its road condition (`flooded` or `dry`), location, and recording start date and time (`yyyymmdd_hhmmss`), and is provided in both a KITTI-style format and the native RTMaps recording format.

```
  ├── flooded                                 # Location sequences captured during flooding events
  │   ├── KITTI-style                         # Sequences in a KITTI-style format
  │   |   ├── Cambogan_20250811_113017        # Sequence by location
  │   |   |   ├── back-imgs
  │   |   |   |   └── <timestamp>.png         # Images in 'png' format
  │   |   |   ├── back-labels
  │   |   |   |   └── <timestamp>.png         # Semantic labels in 'png' format
  │   |   |   ├── front-imgs
  │   |   |   |   └── <timestamp>.png         # Images in 'png' format
  │   |   |   ├── front-labels
  │   |   |   |   └── <timestamp>.png         # Semantic labels in 'png' format
  │   |   |   ├── imu
  │   |   |   |   └── <timestamp>.txt         # IMU data formatted as a 'txt' file
  │   |   |   ├── ouster
  │   |   |   |   └── <timestamp>.bin         # Point clouds formatted as a binary file
  │   |   |   ├── ouster_ground_equations     # Estimated ground plane equations
  │   |   |   ├── ouster_semantic_labels_dry      # Point cloud labels transferred from dry point clouds
  │   |   |   ├── ouster_semantic_labels_flooded  # Point cloud labels from the original flooded point clouds
  │   |   |   └── utm
  │   |   |       └── <timestamp>.txt         # UTM locations formatted as a 'txt' file
  │   |   ├── ...
  │   |   └── ...
  │   └── native-RTmaps                       # Sequences in native recording format
  │       ├── Cambogan_20250811_113017        # Sequence by location
  │       |   ├── Camera_Rec                  # Recording files for image playback
  │       |   ├── IMU_Info_Rec                # Recording files for IMU playback
  │       |   └── Ouster_Rec                  # Recording files for LiDAR playback
  │       ├── ...
  │       └── ...
  │
  └── dry                                 # Location sequences captured while 'dry'
      ├── KITTI-style
      └── native-RTmaps
```

> The native RTMaps format is for direct replay of the vehicle's data capture, but requires RTMaps software (a licence is needed). The KITTI-style format is sampled at ≈ 10 Hz. Data is not synchronised across modalities, but timestamps come from a centralised computer and can be used for time-based alignment.

## Data Formats

### Image Format
Images are stored in PNG format, rectified, at 1920×1200. Using the camera calibration provided, the intrinsics are fx = fy ≈ 2047.8 px (12 mm focal length × 170.648 px/mm) with the principal point at (960, 600).

### Point Cloud Format
Point clouds are stored in binary format (.bin), with each point containing x, y, z positions, as well as reflectivity values. Reflectivity values are surface normalized signal intensity measurements that range from 0 to 255. 3D coordinates are captured in the right-hand coordinate frame with the positive x-axis in the vehicle's direction of travel.

Every point cloud contains 65,536 points in a consistent beam/azimuth order. Points without a valid return (which is common over bodies of water) are recorded with x, y, z and intensity values of 0. The point clouds are released without motion correction; the development kit provides the configuration needed to restructure them into range images.

### UTM Format
UTM data is stored in text file format (.txt), with UTM x and y values being stored as space separated values in the file. Values are in WGS84 / UTM zone 56S.

### IMU Format
Additional IMU information is also stored in text file format (.txt). A space delimiter is again used to separate values. The additional IMU data is stored in the following order:
```
[ Latitude, Longitude, Altitude,
Roll, Pitch, Yaw,
North Velocity, East Velocity,
x Velocity, y Velocity, z Velocity,
x Angular Velocity, y Angular Velocity, z Angular Velocity,
x Angular Velocity, y Angular Velocity, z Angular Velocity,
x Angular Accel, y Angular Accel, z Angular Accel,
x Angular Accel, y Angular Accel, z Angular Accel,
Position Accuracy, Velocity Accuracy,
Navstate Value, Numstat Value,
Position Mode, Velocity Mode, Orientation Mode ]
```

## Benchmarks

FRED includes baseline evaluations of image-based semantic segmentation, LiDAR and sensor-fusion segmentation, and visual place recognition. These are intended to demonstrate the dataset's utility and the open challenges, not to set state-of-the-art results. See the [paper](https://arxiv.org/abs/2605.22018) for full setup details and discussion.

<details>
<summary><b>Image-based water segmentation</b> (water-class mean IoU, %, averaged over all images)</summary>

| Method | Mount Cotton | Cambogan | Dairy Creek | Holmview | Pullenvale |
|---|---:|---:|---:|---:|---:|
| *Pre-trained* | | | | | |
| GA-Nav (RUGD) | 21.8 | 13.6 | 0.0 | 17.0 | 20.8 |
| GA-Nav (RELLIS) | 23.0 | 0.56 | 5.75 | 2.75 | 7.62 |
| V-Flood | 30.3 | 19.4 | 43.4 | 19.8 | 19.5 |
| YOLOv8 | 19.2 | 12.4 | 0.0 | 25.0 | 24.9 |
| *Fine-tuned on Mount Cotton (training), tested on other sequences* | | | | | |
| DeepLab V3 RAU | 86.88 (train) | 31.1 | 6.93 | 18.8 | 18.0 |
| V-Flood | 52.3 (train) | 32.3 | 29.4 | 21.5 | 13.3 |

Existing methods generalise poorly to the varied flood appearances in FRED. Challenges include detecting water at a safe stopping distance and false positives on shadows and non-uniform road surfaces.

</details>

<details>
<summary><b>LiDAR and sensor-fusion water segmentation</b> (water-class BEV IoU, %)</summary>

| Method | Cambogan | Dairy Creek | Holmview | Pullenvale |
|---|---:|---:|---:|---:|
| LiDAR baseline, all points | 10.9 | 8.52 | 21.8 | 14.1 |
| LiDAR baseline, image-bound points | 23.5 | 15.3 | 22.7 | 15.9 |
| Sensor fusion (DeepLab RAU + LiDAR) | 27.7 | 1.64 | 39.5 | 25.6 |

All rows are averaged over all samples. The LiDAR baseline completes missing ground returns by projecting void rays onto an estimated ground plane. It improves as the vehicle approaches the water.

</details>

<details>
<summary><b>Visual place recognition</b> (Recall@1 at 10 m tolerance, flooded query sequences vs. dry reference)</summary>

| Descriptor | Avg. reduction (dry → flooded) |
|---|---:|
| BoQ | 4.83 |
| Clique Mining | 5.53 |
| CosPlace | 6.62 |
| CricaVPR | 18.23 |
| EigenPlaces | 7.18 |
| MixVPR | 6.65 |
| SALAD | 6.42 |
| SuperVLAD | 8.03 |

All descriptors reach 100% Recall@1 on the dry query sequences at this tolerance. Flooded roads reduce performance, most strongly at Cambogan.

</details>

## Citation

If you use this dataset or development kit in your work, please cite our preprint:

```bibtex
@article{malone2026fred,
  title         = {FRED: A Multi-Modal Autonomous Driving Dataset for Flooded Road Environments},
  author        = {Malone, Connor and Demmel, S{\'e}bastien and Glaser, S{\'e}bastien},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2605.22018},
  eprint        = {2605.22018},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.22018}
}
```

## Licence

- **Dataset**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Development kit (code)**: [BSD 3-Clause](LICENSE).

## Acknowledgements

This work was supported by the Queensland Department of Transport and Main Roads, the iMove Cooperative Research Centre (CRC) through the project *Expanding Operating Design Domain of Automated Vehicles* (project 1-075), and the Australian Research Council (ARC) Industrial Transformation Training Centre for Automated Vehicles in Rural and Remote Regions (IC230100001). We thank Amit Trivedi (Queensland Department of Transport and Main Roads), Fatemeh Ghorbani (QUT) and Long Wang (QUT) for their contributions.

## Contact

Connor Malone — [cj.malone@qut.edu.au](mailto:cj.malone@qut.edu.au), Queensland University of Technology.  
For bugs, questions and feature requests, please open a [GitHub issue](https://github.com/AVR3-Training-Centre/python-FRED/issues).
