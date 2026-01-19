from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# def gaussian_mask(
#     gaze: torch.Tensor, sigma: float, shape: Tuple[int, int]
# ) -> torch.Tensor:
#     """
#     Generates a single 2D Gaussian mask for a specific gaze point.
#
#     :param gaze: (2). Coordinates of gaze point (normalized 0-1).
#     :param sigma: Standard deviation (spread) of the Gaussian.
#     :param shape: (height, width) of the output mask.
#     :return: torch.Tensor: A 2D tensor of the specified shape.
#     """
#     h, w = shape
#     device = gaze.device
#
#     x0, y0 = gaze
#
#     x0 = x0 * w
#     y0 = y0 * h
#
#     x_grid = torch.arange(0, w, dtype=torch.float32, device=device)
#     y_grid = torch.arange(0, h, dtype=torch.float32, device=device)
#     x, y = torch.meshgrid(x_grid, y_grid, indexing="xy")
#
#     dist_sq = (x - x0) ** 2 + (y - y0) ** 2
#
#     mask = torch.exp(-dist_sq / (2.0 * sigma**2))
#
#     return mask


def decaying_gaussian_mask(
    gaze_coords: torch.Tensor,
    shape: Tuple[int, int],
    sigma: float = 15,  # gamma in the paper
    decay: float = 0.99,  # beta in the paper
    alpha: float = 0.7,  # alpha in the paper
) -> torch.Tensor:
    """
    Generates cumulative heatmaps with coordinate smoothing (Alpha) and
    temporal decay (Beta).

    :param gaze_coords: (..., layers, 2). Last dim is (x, y), 2nd-to-last is Time.
    :param shape: (height, width) of the image.
    :param sigma: Spread of the gaussian (Gamma).
    :param decay: Rate at which previous mask fades (Beta).
    :param alpha: Smoothing factor for coordinates (0 = no smoothing, 1 = static).
    :return: (..., height, width). Batch dims matching input.
    """
    H, W = shape
    device = gaze_coords.device

    batch_shape = gaze_coords.shape[:-2]
    layers = gaze_coords.shape[-2]

    gaze_flat = gaze_coords.view(-1, layers, 2).clone()
    B, L, _ = gaze_flat.shape

    # current = alpha * prev + (1 - alpha) * raw
    if alpha > 0:
        for t in range(1, L):
            prev_coords = gaze_flat[:, t - 1, :]
            curr_coords = gaze_flat[:, t, :]

            prev_valid = ~torch.isnan(prev_coords).any(dim=1)
            curr_valid = ~torch.isnan(curr_coords).any(dim=1)
            should_smooth = prev_valid & curr_valid

            if should_smooth.any():
                gaze_flat[should_smooth, t, :] = (
                    alpha * prev_coords[should_smooth]
                    + (1 - alpha) * curr_coords[should_smooth]
                )

    x_range = torch.arange(0, W, dtype=torch.float32, device=device)
    y_range = torch.arange(0, H, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="xy")

    grid_x = grid_x.unsqueeze(0)  # (1, H, W)
    grid_y = grid_y.unsqueeze(0)  # (1, H, W)

    heatmap = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    denom = 2.0 * sigma**2

    for i in range(layers):
        heatmap = heatmap * decay

        current_points = gaze_flat[:, i, :]

        valid_mask = ~torch.isnan(current_points).any(dim=1)

        x0 = (current_points[:, 0] * W).view(B, 1, 1)
        y0 = (current_points[:, 1] * H).view(B, 1, 1)

        dist_sq = (grid_x - x0) ** 2 + (grid_y - y0) ** 2
        current_masks = torch.exp(-dist_sq / denom)

        current_masks = current_masks * valid_mask.view(B, 1, 1)

        heatmap = torch.maximum(heatmap, current_masks)

    return heatmap.view(*batch_shape, H, W)


def patchify(gaze_mask: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Splits an image/mask into non-overlapping patches.

    :param gaze_mask: (..., H, W).
    :param patch_size: (Patch_H, Patch_W) in pixels.
    :return: (..., Grid_Rows, Grid_Cols, Patch_H, Patch_W).
    """
    patch_h, patch_w = patch_size

    batch_shape = gaze_mask.shape[:-2]
    H, W = gaze_mask.shape[-2:]

    assert H % patch_h == 0 and W % patch_w == 0, (
        f"Image size ({H},{W}) must be divisible by patch size ({patch_h},{patch_w})"
    )

    num_patches_h = H // patch_h
    num_patches_w = W // patch_w

    gaze_mask = gaze_mask.reshape(-1, H, W)
    B, _, _ = gaze_mask.shape

    gaze_mask = gaze_mask.view(B, num_patches_h, patch_h, num_patches_w, patch_w)

    gaze_mask = gaze_mask.permute(0, 1, 3, 2, 4)

    return gaze_mask.view(*batch_shape, num_patches_h, num_patches_w, patch_h, patch_w)


def plot_patches(patches: torch.Tensor, spacing: int = 1):
    """
    Visualizes patches using PyTorch operations.

    :param patches: (Grid_Rows, Grid_Cols, Patch_H, Patch_W)
    :param spacing: Spacing between patches in pixels.
    """
    if patches.ndim != 4:
        raise ValueError(f"Patches must be 4D, got {patches.shape}")

    patches = patches.detach().cpu()

    rows, cols, ph, pw = patches.shape

    full_h = (rows * ph) + ((rows - 1) * spacing)
    full_w = (cols * pw) + ((cols - 1) * spacing)

    canvas = torch.full((full_h, full_w), float("nan"), dtype=patches.dtype)

    for i in range(rows):
        for j in range(cols):
            y_start = i * (ph + spacing)
            x_start = j * (pw + spacing)

            canvas[y_start : y_start + ph, x_start : x_start + pw] = patches[i, j]

    plt.figure(figsize=(8, 8))

    cmap = plt.cm.hot
    cmap.set_bad(color="grey")

    plt.imshow(canvas.numpy(), cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Layout: {rows}x{cols} | Patch Size: {ph}x{pw}")
    plt.show()


# mask = decaying_gaussian_mask([(0.5, 0.5), (0.4, 0.6), (1, 0.5)], 5, (84, 84), 0.5)
#
# patched_mask = patchify(mask, patch_size=(6, 6))
# plot_patches(patched_mask, spacing=1)
