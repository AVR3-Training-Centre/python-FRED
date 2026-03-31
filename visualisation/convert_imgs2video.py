#!/usr/bin/env python3
"""
images_to_mp4.py — Combine a directory of images into an MP4 video.

Usage:
    python images_to_mp4.py <image_dir> [options]

Options:
    --fps FPS           Frames per second (default: 10)
    --output OUTPUT     Output file path (default: output.mp4)
    --pattern PATTERN   Glob pattern for images (default: auto-detect)
    --sort {name,time}  Sort order for frames (default: name)
"""

import argparse
import sys
from pathlib import Path
from natsort import natsorted
import cv2


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(directory: Path, pattern: str | None, sort: str) -> list[Path]:
    if pattern:
        images = sorted(directory.glob(pattern))
    else:
        images = [
            p for p in directory.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if sort == "time":
            images.sort(key=lambda p: p.stat().st_mtime)
        else:
            images.sort(key=lambda p: p.name)

    return natsorted(images)


def build_video(images: list[Path], output: Path, fps: int) -> None:
    # Read the first frame to get dimensions
    first = cv2.imread(str(images[0]))
    if first is None:
        sys.exit(f"Error: could not read image '{images[0]}'")

    height, width = first.shape[:2]
    print(f"Frame size : {width}x{height}")
    print(f"Frame count: {len(images)}")
    print(f"FPS        : {fps}")
    print(f"Output     : {output}")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fallback: try mp4v codec if avc1 unavailable
        print("Warning: avc1 (H.264) unavailable, falling back to mp4v codec.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    if not writer.isOpened():
        sys.exit("Error: could not open VideoWriter. Check OpenCV build & codecs.")

    for i, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  Warning: skipping unreadable file '{path.name}'")
            continue
        # Resize if a frame differs from the first frame's dimensions
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        if i % 50 == 0 or i == len(images):
            print(f"  Encoded {i}/{len(images)} frames...")

    writer.release()
    print(f"\nDone! Video saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine images in a directory into an MP4 video."
    )
    parser.add_argument("image_dir", help="Path to directory containing images")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--output", default="output.mp4", help="Output video file (default: output.mp4)")
    parser.add_argument("--pattern", default=None, help="Glob pattern, e.g. 'frame_*.png'")
    parser.add_argument("--sort", choices=["name", "time"], default="name",
                        help="Sort frames by name (default) or modification time")
    args = parser.parse_args()

    directory = Path(args.image_dir)
    if not directory.is_dir():
        sys.exit(f"Error: '{directory}' is not a valid directory.")

    images = collect_images(directory, args.pattern, args.sort)
    if not images:
        sys.exit(f"Error: no supported images found in '{directory}'.")

    print(f"Found {len(images)} image(s) in '{directory}'")
    build_video(images, Path(args.output), args.fps)


if __name__ == "__main__":
    main()