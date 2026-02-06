import math
import os
import platform
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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
    folder: str, frame_stack: int, gaze_temporal_decay: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load Atari data for training and testing.

    :param folder: The folder of the Atari game dataset.
    :param frame_stack: The number of frames to stack per episode.
    :param gaze_temporal_decay: The gaze temporal decay, used to calculate
        the amount of stacking layers (until opacity = 5%).
    :param device: The device to use.
    :return: (B, F, C, H, W), (B, F, layers, 2), (B)
    """
    files_list = [p for p in Path(folder).iterdir() if p.is_file()]
    dataset = torch.load(files_list[0], weights_only=False)

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

    layers = int(math.ceil(math.log(0.005, gaze_temporal_decay)))
    gaze_coord_list = layer_gazes(gaze_coord_list, layers=layers)

    observations, gaze_coords = stack_observations_and_gaze_coords(
        observation_list, gaze_coord_list, frame_stack
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


MAX_EPISODES = {
    "Alien": 20,
    "Asterix": 20,
    "Assault": 20,
    "Breakout": 20,
    "ChopperCommand": 20,
    "DemonAttack": 20,
    "Enduro": 20,
    "Frostbite": 20,
    "Freeway": 20,
    "MsPacman": 20,
    "Phoenix": 20,
    "Qbert": 20,
    "RoadRunner": 20,
    "Seaquest": 50,
    "UpNDown": 20,
}

MAX_EPISODES_ATARI_HEAD = {
    "Seaquest": 20,
    "MsPacman": 20,
    "Enduro": 20,
    "Freeway": 10,
    "Hero": 20,
    "BankHeist": 20,
}  # Atari_Head data


def get_font(size=16):
    system = platform.system()

    if system == "Darwin":  # macOS
        paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    elif system == "Linux":
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        ]
    else:  # Windows
        return ImageFont.truetype("arial.ttf", size)

    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    # Fallback if nothing is found
    return ImageFont.load_default()


class FrameWriter:
    def __init__(self, input_dim, num_actions):
        self.input_dim = input_dim

        self.num_actions = num_actions

        font = get_font(16)

        pre_rendered_frames = []
        for action in range(num_actions):
            text_img = Image.new("RGBA", (84, 84), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_img)

            text = str(action)
            draw.text((11, 55), text, font=font, fill=(255, 255, 255, 255))
            text_tensor = torch.from_numpy(np.array(text_img))
            text_tensor = text_tensor[:, :, :3] * (text_tensor[:, :, 3:] / 255)
            text_tensor = text_tensor[:, :, 0]
            # print(text_tensor.shape, text_tensor.max(), text_tensor.min())
            if self.input_dim != (84, 84):
                text_tensor = torchvision.transforms.functional.resize(
                    text_tensor[None], self.input_dim
                )[0]
            pre_rendered_frames.append(text_tensor)

        pre_rendered_frames = torch.stack(pre_rendered_frames).numpy().astype(np.uint16)

        self.pre_rendered_frames = pre_rendered_frames

    def add_text_tensor_to_frame(self, frame, action=None, channel_first=True):
        if action is None:
            return frame

        action_frame = self.pre_rendered_frames[action]
        frame = frame.astype(np.uint16)

        if channel_first:
            addable_frame = action_frame[None]
        else:
            addable_frame = action_frame[:, :, None]

        return np.clip(frame + addable_frame, 0, 255).astype(np.uint8)


class GazeToMask:
    def __init__(self, N=84, sigmas=[10, 10, 10, 10], coeficients=[1, 1, 1, 1]):
        self.N = N
        assert len(sigmas) == len(coeficients)
        self.sigmas = sigmas
        self.coeficients = coeficients
        self.masks = self.initialize_mask()

    def generate_single_gaussian_tensor(self, map_size, mean_x, mean_y, sigma):
        x = (
            torch.arange(map_size, dtype=torch.float32)
            .unsqueeze(1)
            .expand(map_size, map_size)
        )
        y = (
            torch.arange(map_size, dtype=torch.float32)
            .unsqueeze(0)
            .expand(map_size, map_size)
        )
        # Calculate the Gaussian distribution for each element
        gaussian_tensor = (1 / (2 * torch.pi * sigma**2)) * torch.exp(
            -((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma**2)
        )

        return gaussian_tensor

    def initialize_mask(self):
        temp_map = []
        N = self.N
        for i in range(len(self.sigmas)):
            temp = self.generate_single_gaussian_tensor(
                2 * N, N - 1, N - 1, self.sigmas[i]
            )
            temp = temp / temp.max()
            temp_map.append(self.coeficients[i] * temp)

        temp_map = torch.stack(temp_map, 0)

        return temp_map

    def find_suitable_map(self, Nx2=168, index=0, mean_x=0.5, mean_y=0.5):
        # returns a map such that the center of the gaussian is located at (mean_x, mean_y) of the map
        start_x, start_y = int((1 - mean_x) * Nx2 / 2), int((1 - mean_y) * Nx2 / 2)
        desired_map = self.masks[index][
            start_y : start_y + Nx2 // 2, start_x : start_x + Nx2 // 2
        ]
        return desired_map

    def find_bunch_of_maps(self, means=[[0.5, 0.5]], offset_start=0):
        current_maps = torch.zeros([self.N, self.N])
        bunch_size = len(means)
        assert bunch_size + offset_start <= len(self.sigmas), (
            f"The bunch is too long! It's length is {bunch_size}"
        )
        Nx2 = self.N * 2
        for i in range(bunch_size):
            mean_x, mean_y = means[i][0], means[i][1]
            # mean_x, mean_y = 0, 0
            temp = self.find_suitable_map(Nx2, i + offset_start, mean_x, mean_y)
            current_maps = current_maps + temp

        return current_maps / torch.max(current_maps)


def load_dataset(
    env,
    seed,
    datapath,
    conf_type,
    conf_randomness,
    stack,
    num_episodes=None,
    use_gaze=False,
    data_source="Our",
    gaze_mask_sigma=15.0,
    gaze_mask_coef=0.7,
):
    if num_episodes:
        if data_source == "Atari_HEAD":
            assert num_episodes <= MAX_EPISODES_ATARI_HEAD[env], (
                f"The number of available episodes is {MAX_EPISODES_ATARI_HEAD[env]}, but {num_episodes} is requested"
            )
        else:
            assert num_episodes <= MAX_EPISODES[env], (
                f"The number of available episodes is {MAX_EPISODES[env]}, but {num_episodes} is requested"
            )

    # load the data
    path = os.path.join(datapath, env)
    file_name = f"/num_episodes_{MAX_EPISODES[env]}_fs4_human.pt"
    atari_head_file_name = f"Atari_Head_Data/{env}/ordinary.pt"
    loaded_obj = (
        torch.load(atari_head_file_name)
        if data_source == "Atari_HEAD"
        else torch.load(path + file_name, weights_only=False)
    )

    obs = loaded_obj["observations"]
    rewards = loaded_obj["episode-rewards"]
    actions = loaded_obj["actions"]
    episode_lengths = loaded_obj["steps"]
    truncateds = loaded_obj["truncateds"]
    terminateds = loaded_obj["terminateds"]

    gaze_info = loaded_obj["gaze_information"] if use_gaze else None

    assert num_episodes <= episode_lengths.shape[0], (
        "num_episodes should be less than total saved episodes"
    )

    obs = torch.from_numpy(obs)
    actions = torch.from_numpy(actions)
    episode_lengths = torch.from_numpy(episode_lengths)
    truncateds = torch.from_numpy(truncateds)
    terminateds = torch.from_numpy(terminateds)

    if use_gaze:
        gaze_info = torch.from_numpy(gaze_info)

        g = gaze_info[:, :2]
        g[g < 0] = 0
        g[g > 1] = 1

        gaze_info[:, :2] = g

    short_memory_length = 20
    stride = 2

    saliency_sigmas = [
        gaze_mask_sigma / (0.99 ** (short_memory_length - i))
        for i in range(short_memory_length + 1)
    ]
    coeficients = [
        gaze_mask_coef ** (short_memory_length - i)
        for i in range(short_memory_length + 1)
    ]
    coeficients += coeficients[::-1][1:]
    saliency_sigmas += saliency_sigmas[::-1][1:]

    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)

    episode_obs = []
    episode_actions = []

    episode_gaze = []

    for episode, _ in enumerate(episode_lengths):
        start = sum(episode_lengths[:episode])
        end = start + episode_lengths[episode]
        episode_obs.append(obs[start + episode : end + episode + 1])

        episode_actions.append(actions[start:end])
        if use_gaze:
            episode_gaze.append(gaze_info[start:end])
        assert terminateds[end - 1] or truncateds[end - 1]

    if num_episodes:
        episode_obs = episode_obs[:num_episodes]
        episode_actions = episode_actions[:num_episodes]
        if use_gaze:
            episode_gaze = episode_gaze[:num_episodes]

    episode_obs = [ep[:-1] for ep in episode_obs]
    if use_gaze:
        episode_saliency_gaze = [
            torch.stack(
                [
                    MASK.find_bunch_of_maps(
                        means=ep[
                            max(0, j - stride * short_memory_length) : min(
                                short_memory_length * stride + j + 1, len(ep)
                            ) : stride
                        ],
                        offset_start=max(short_memory_length - j, 0),
                    )
                    for j in range(len(ep))
                ],
                0,
            )
            for ep in tqdm(episode_gaze)
        ]
        episode_gaze_coordinates = [ep[:, :2] for ep in episode_gaze]

    else:
        episode_saliency_gaze = [
            torch.zeros_like(episode_obs[i], dtype=torch.float32)
            for i in range(len(episode_obs))
        ]
        episode_gaze_coordinates = [
            torch.zeros((len(episode_obs[i]), 2), dtype=torch.float32)
            for i in range(len(episode_obs))
        ]

    # repeat the first frame for stack - 1 times
    episode_obs = [
        torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep]) for ep in episode_obs
    ]
    episode_saliency_gaze = [
        torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep])
        for ep in episode_saliency_gaze
    ]

    for i, (ep_obs, ep_gaze) in enumerate(zip(episode_obs, episode_saliency_gaze)):
        new_episode_obs = []
        new_episode_saliency_gaze = []
        for s in range(stack):
            end = None if s == stack - 1 else s - stack + 1
            new_episode_obs.append(ep_obs[s:end])
            new_episode_saliency_gaze.append(ep_gaze[s:end])

        episode_obs[i] = torch.stack(new_episode_obs, dim=1)
        episode_saliency_gaze[i] = torch.stack(new_episode_saliency_gaze, dim=1)

    unique_actions = np.unique(actions)
    num_actions = unique_actions.max() + 1
    fw = FrameWriter((84, 84), num_actions)

    if conf_type != "normal":
        print("Building dataset by writing actions over the images..")

    rnd_generator = np.random.default_rng(seed)

    for i, ep in tqdm(enumerate(episode_obs), total=len(episode_obs)):
        new_obs = []
        for j in range(1, len(ep)):
            if conf_type == "confounded":
                action_to_write = episode_actions[i][j - 1].item()
            elif conf_type == "normal":
                action_to_write = None
            else:
                raise NotImplementedError(conf_type)

            if conf_type in ["confounded"]:
                if rnd_generator.random() < conf_randomness:
                    action_to_write = rnd_generator.integers(num_actions)

            new_obs.append(fw.add_text_tensor_to_frame(ep[j].numpy(), action_to_write))

        episode_obs[i] = torch.from_numpy(np.stack(new_obs))

    episode_actions = [ep[1:] for ep in episode_actions]
    episode_saliency_gaze = [ep[1:] for ep in episode_saliency_gaze]
    episode_gaze_coordinates = [ep[1:] for ep in episode_gaze_coordinates]

    observations = torch.cat(episode_obs)
    actions = torch.cat(episode_actions)
    gaze_saliency_maps = torch.cat(episode_saliency_gaze)
    gaze_coordinates = torch.cat(episode_gaze_coordinates)

    return observations, actions, gaze_saliency_maps, gaze_coordinates
