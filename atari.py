import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import dtype


def list_games(path: str) -> List[str]:
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]


def break_episodes(
    observations: torch.Tensor, gaze_coords: torch.Tensor, terminals: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Breaks observations and gaze coordinates by episodes.

    :param observations: (N, H, W). Raw observations from dataset.
    :param gaze_coords: (N, 2). Raw gaze coordinates from dataset.
    :param terminals: (N). Episode boundaries from dataset.
    :return: List[(ep_len, H, W)], List[(ep_len, 2)]
    """
    term_indices = torch.where(terminals)[0].tolist()

    obs_episodes = []
    gaze_episodes = []

    gaze_ptr = 0
    obs_ptr = 0

    for term_idx in term_indices:
        episode_len = (term_idx + 1) - gaze_ptr

        ep_gaze = gaze_coords[gaze_ptr : gaze_ptr + episode_len]

        ep_obs = observations[obs_ptr : obs_ptr + episode_len]

        obs_episodes.append(ep_obs)
        gaze_episodes.append(ep_gaze)

        gaze_ptr += episode_len
        obs_ptr += episode_len + 1

    return obs_episodes, gaze_episodes


def layer_gazes(
    gaze_coords: List[torch.Tensor], layers: int = 20
) -> List[torch.Tensor]:
    """
    Stacks the previous l-1 gaze coordinates for each time step.
    If there aren't enough previous frames (at the start), repeats the first frame.

    :param gaze_coords: List of tensors, each shape (N, 2)
    :param layers: History layers to stack
    :return: List of tensors, each shape (N, layers, 2)
    """
    layered_gazes = []

    for episode_gaze in gaze_coords:
        if layers > 1:
            padding = episode_gaze[0].unsqueeze(0).repeat(layers - 1, 1)
            padded_gaze = torch.cat([padding, episode_gaze], dim=0)
        else:
            padded_gaze = episode_gaze

        windows = padded_gaze.unfold(0, layers, 1)
        windows = windows.permute(0, 2, 1)
        layered_gazes.append(windows)

    return layered_gazes


def stack_observations_and_gaze_coords(
    observation_list: List[torch.Tensor], gaze_coord_list: List[torch.Tensor], k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    processed_obs = []
    processed_gcs = []

    def stack(t: torch.Tensor) -> torch.Tensor:
        first_frame = t[0]
        padding = first_frame.unsqueeze(0).repeat(k - 1, 1, 1)

        padded = torch.cat([padding, t], dim=0)
        windows = padded.unfold(0, k, 1)
        stacked = windows.permute(0, 3, 1, 2)

        return stacked

    for ep_obs, ep_gc in zip(observation_list, gaze_coord_list):
        if ep_obs.ndim != 3:
            raise ValueError(f"Expected (N, H, W), got {ep_obs.shape}")

        if ep_gc.ndim != 3:
            raise ValueError(f"Expected (N, layers, 2), got {ep_gc.shape}")

        processed_obs.append(stack(ep_obs))
        processed_gcs.append(stack(ep_gc))

    return (
        torch.cat(processed_obs, dim=0),
        torch.cat(processed_gcs, dim=0),
    )


def load_data(
    path: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load Atari data for training and testing.

    :param path: The path of the Atari game dataset.
    :param device: The device to use.
    :return: (B, F, C, H, W), (B, F, layers, 2), (B)
    """
    dataset = torch.load(path, weights_only=False)

    observations = torch.from_numpy(dataset["observations"]).to(
        dtype=torch.float, device=device
    )
    actions = torch.from_numpy(dataset["actions"]).to(dtype=torch.long, device=device)
    terminals = torch.from_numpy(dataset["terminateds"]).to(
        dtype=torch.bool, device=device
    )
    gaze_coords = torch.from_numpy(dataset["gaze_information"]).to(
        dtype=torch.float, device=device
    )
    gaze_coords = gaze_coords[:, :2]

    observations = observations / 255.0

    observation_list, gaze_coord_list = break_episodes(
        observations, gaze_coords, terminals
    )
    gaze_coord_list = layer_gazes(gaze_coord_list, layers=20)
    observations, gaze_coords = stack_observations_and_gaze_coords(
        observation_list, gaze_coord_list, 4
    )

    return observations.unsqueeze(2), gaze_coords, actions


def plot_frames(frames: torch.Tensor):
    """
    Plots frames from a (F, C, H, W) tensor in a square grid.

    :param frames: (F, C, H, W). Values should be roughly in [0, 1] for floats or [0, 255] for uint8.
    """
    frames = frames.detach().cpu().numpy()

    if frames.ndim != 4:
        raise ValueError(f"Expected shape (F, C, H, W), got {frames.shape}")

    F, C, H, W = frames.shape

    cols = math.ceil(math.sqrt(F))
    rows = math.ceil(F / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]

        if i < F:
            img = frames[i]

            img = np.transpose(img, (1, 2, 0))

            if C == 1:
                ax.imshow(img.squeeze(-1))
            else:
                ax.imshow(img)

            ax.set_title(f"Frame {i}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()
