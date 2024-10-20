"""
Design pattern of these transforms:
1. Take in dictionary of sample data
2. Look for specific inputs in the sample
3. Process the inputs
4. Add new data to the sample
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as tv_transforms
import scipy.ndimage

from .road_utils import Track


def project(points, view, proj, h, w):
    points_uv_raw = points @ view @ proj
    points_uv = points_uv_raw / points_uv_raw[:, -1:]

    # convert from uv to pixel coordinates, [0, W] and [0, H]
    points_img = points_uv[:, :2]
    points_img[:, 0] = (points_img[:, 0] + 1) * w / 2
    points_img[:, 1] = (1 - points_img[:, 1]) * h / 2

    mask = (
        (points_uv_raw[:, -1] > 1)  # must be in front of camera
        & (points_uv_raw[:, -1] < 15)  # don't render too far
        & (points_img[:, 0] >= 0)  # projected in valid img width
        & (points_img[:, 0] < w)
        & (points_img[:, 1] >= 0)  # projected in valid img height
        & (points_img[:, 1] < h)
    )

    return points_img[mask], mask


def rasterize_lines(
    points: np.ndarray,
    canvas: np.ndarray,
    color: int,
    thickness: int = 4,
):
    for i in range(len(points) - 1):
        start = points[i].astype(int)
        end = points[i + 1].astype(int)

        cv2.line(canvas, tuple(start), tuple(end), color, thickness)


class Compose(tv_transforms.Compose):
    def __call__(self, sample: dict):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ImageLoader:
    def __init__(self, episode_path: str):
        self.episode_path = Path(episode_path)

    def __call__(self, sample: dict):
        image_path = self.episode_path / f"{sample['_idx']:05d}_im.jpg"
        image = np.uint8(Image.open(image_path)) / 255.0
        image = image.transpose(2, 0, 1)

        sample["image"] = image.astype(np.float32)

        return sample


class DepthLoader(ImageLoader):
    def __call__(self, sample: dict):
        depth_path = self.episode_path / f"{sample['_idx']:05d}_depth.png"
        depth = np.uint16(Image.open(depth_path)) / 65535.0

        sample["depth"] = depth.astype(np.float32)

        return sample



class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        angle = np.random.uniform(-self.degrees, self.degrees)
        sample["image"] = scipy.ndimage.rotate(sample["image"], angle, axes=(1, 2), reshape=False, mode='nearest')
        sample["depth"] = scipy.ndimage.rotate(sample["depth"], angle, axes=(0, 1), reshape=False, mode='nearest')
        sample["track"] = scipy.ndimage.rotate(sample["track"], angle, axes=(0, 1), reshape=False, mode='nearest', order=0)
        return sample


class RandomVerticalFlip(tv_transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample["image"] = np.flip(sample["image"], axis=1).copy()
            sample["depth"] = np.flip(sample["depth"], axis=0).copy()
            # Flip track labels if necessary
            flip_track = np.flip(sample["track"], axis=0).copy()
            sample["track"] = np.select([flip_track == 1, flip_track == 2], [2, 1], flip_track)
        
        return sample


class RandomResizedCrop(tv_transforms.RandomResizedCrop):
    def __init__(self, size=(96, 128), scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, sample):
        # Get original dimensions
        h, w = sample["image"].shape[1:3]
        area = h * w

        # Randomly pick the target area
        target_area = np.random.uniform(self.scale[0] * area, self.scale[1] * area)
        aspect_ratio = np.random.uniform(0.75, 1.33)  # Aspect ratio range

        # Calculate the target height and width
        h_target = int(round(np.sqrt(target_area * aspect_ratio)))
        w_target = int(round(np.sqrt(target_area / aspect_ratio)))

        # Randomly crop the image
        top = np.random.randint(0, h - h_target + 1)
        left = np.random.randint(0, w - w_target + 1)

        sample["image"] = sample["image"][:, top:top + h_target, left:left + w_target]
        sample["depth"] = sample["depth"][top:top + h_target, left:left + w_target]
        sample["track"] = sample["track"][top:top + h_target, left:left + w_target]

        # Resize to the desired output size
        sample["image"] = F.resize(sample["image"], self.size)
        sample["depth"] = F.resize(sample["depth"], self.size)
        sample["track"] = F.resize(sample["track"], self.size)

        return sample


class RandomHorizontalFlip(tv_transforms.RandomHorizontalFlip):
    def __call__(self, sample: dict):
        if np.random.rand() < self.p:
            sample["image"] = np.flip(sample["image"], axis=2).copy()
            # Flip label assignment
            flip_track = np.flip(sample["track"], axis=1).copy()
            sample["track"] = np.select([flip_track == 1, flip_track == 2], [2, 1], flip_track)
            sample["depth"] = np.flip(sample["depth"], axis=1).copy()

        return sample


class TrackProcessor:
    def __init__(self, track: Track):
        self.track = track

    def __call__(self, sample: dict):
        frames = sample["_frames"]
        idx = sample["_idx"]
        h, w = sample["image"].shape[1:]

        loc = frames["loc"][idx].copy()
        proj = frames["P"][idx].copy()
        view = frames["V"][idx].copy()
        view[-1, :3] += -1.0 * view[1, :3]  # move camera slightly

        # project track points into image space
        track_left, _ = project(self.track.track_left, view, proj, h, w)
        track_right, _ = project(self.track.track_right, view, proj, h, w)

        # draw line segments onto a blank canvas
        track = np.zeros((h, w), dtype=np.uint8)
        rasterize_lines(track_left, track, color=1)
        rasterize_lines(track_right, track, color=2)

        sample["track"] = track.astype(np.int64)

        return sample
