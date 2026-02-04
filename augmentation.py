import random
from typing import Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as Fn

# def random_noise(
#     x: torch.Tensor, gaze: torch.Tensor = None, std: float = 0.1
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Adds Gaussian noise to x.
#
#     If gaze is provided:
#         - Areas with Gaze=1.0 get 0% noise (Clear).
#         - Areas with Gaze=0.0 get 100% noise (Noisy).
#
#     :param x: (B, F, C, H, W)
#     :param gaze: (B, F, H, W)
#     :param std: Standard deviation of the Gaussian noise.
#     :return: (B, F, C, H, W), (B, F, H, W)
#     """
#     device = x.device
#
#     noise = torch.randn_like(x, device=device) * std
#
#     if gaze is not None:
#         noise_scale = 1.0 - gaze.unsqueeze(2)
#
#         x_noisy = x + (noise * noise_scale)
#     else:
#         x_noisy = x + noise
#
#     x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
#
#     return x_noisy, gaze


class RandomFrameDropout:
    def __init__(self, f=4):
        self.f = f
        self.current_idx = 0
        self.target_idx = random.randint(0, self.f - 1)

    def reset(self):
        self.current_idx = 0
        self.target_idx = random.randint(0, self.f - 1)

    def __call__(self, img, **kwargs):
        ret_img = img.copy()
        if self.current_idx == self.target_idx:
            ret_img[:] = 0

        self.current_idx += 1
        if self.current_idx >= self.f:
            self.reset()

        return ret_img


class Augment:
    def __init__(
        self,
        frame_shape: Tuple[int, int, int, int],
        crop_padding: int,
        light_intensity: float,
        noise_std: float,
        p_pixel_dropout: float,
        posterize_bits: int,
        blur_pixels: int,
        p_spatial_corruption: float,
        p_temporal_corruption: float,
    ):
        F, H, W, C = frame_shape

        crop = A.Compose(
            [
                A.Pad(padding=crop_padding, p=1),
                A.RandomCrop(84, 84, p=1),
            ],
            p=p_spatial_corruption,
        )

        light = A.Compose(
            [
                A.RandomGamma(
                    gamma_limit=(
                        100 * (1 - light_intensity),
                        100 * (1 + light_intensity),
                    ),
                    p=1,
                ),
                A.RandomBrightnessContrast(p=1),
            ],
            p=p_spatial_corruption,
        )

        noise = A.GaussNoise(std_range=(noise_std, noise_std), p=1)

        pixel_drop = A.PixelDropout(dropout_prob=p_pixel_dropout, p=1)

        posterize = A.Posterize(num_bits=posterize_bits, p=1)

        blur = A.GaussianBlur(blur_limit=(blur_pixels, blur_pixels), p=1)

        frame_drop = A.Lambda(image=RandomFrameDropout(F), name="frame_drop", p=1)

        temporal_corruptions = [frame_drop]

        self.augment = A.Compose(
            [
                crop,
                light,
                A.OneOf(
                    [noise],
                    p=p_spatial_corruption,
                ),
            ],
        )

    def __call__(
        self, observations: torch.Tensor, gaze_masks: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param obs: (B, F, C, H, W)
        :param gaze: (B, F, H, W)
        :param kwargs:
        :return: (B, F, C, H, W), (B, F, H, W)
        """
        observations = observations.permute(0, 1, 3, 4, 2).numpy()  # (B, F, H, W, C)
        gaze_masks = gaze_masks.unsqueeze(-1).numpy()  # (B, F, H, W, C)

        B, F, H, W, C = observations.shape

        aug_frames, aug_masks = [], []
        for i in range(B):
            frames = observations[i]
            masks = gaze_masks[i]
            augmented = self.augment(images=frames, masks=masks)
            aug_frames.append(augmented["images"])
            aug_masks.append(augmented["masks"])

        observations = torch.from_numpy(np.stack(aug_frames)).permute(
            0, 1, 4, 2, 3
        )  # (B, F, C, H, W)
        gaze_masks = torch.from_numpy(np.stack(aug_masks)).squeeze(-1)  # (B, F, H, W)

        return observations, gaze_masks
