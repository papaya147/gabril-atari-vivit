import datetime
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import albumentations as A
import ale_py
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
import torch.optim as optim
import wandb
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

import dataset
import gaze
from augmentation import Augment
from device import device
from vivit import AuxGazeFactorizedViViT, FactorizedViViT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Config:
    # paths and flags
    game_index: int = 0
    game: str = ""
    atari_dataset_folder: str = "../atari-dataset"
    use_plots: bool = False
    save_folder: str = "./models"
    seed: int = 42
    algorithm: str = "AuxGazeFactorizedViViT"

    frame_stack: int = 4

    # gaze
    gaze_sigma: int = 15
    gaze_beta: float = 0.99
    gaze_alpha: float = 0.7

    # augmentation
    augment_crop_padding: int = 4
    augment_light_intensity: float = 0.2
    augment_noise_std: float = 0.01
    augment_p_pixel_dropout: float = 0.01
    augment_posterize_bits: int = 4
    augment_blur_pixels: int = 2
    augment_p_spatial_corruptions: float = 0.5
    augment_p_temporal_corruptions: float = 0.25

    # transformer arch
    spatial_patch_size: Tuple[int, int] = (6, 6)
    embedding_dim: int = 64
    spatial_depth: int = 3
    temporal_depth: int = 2
    spatial_heads: int = 4
    temporal_heads: int = 4
    inner_dim: int = 32
    mlp_dim: int = 256
    dropout: float = 0.25

    # hyperparams
    learning_rate: float = 5e-4
    epochs: int = 2000
    train_pct: float = 0.8
    batch_size: int = 32
    lambda_gaze: float = 0.1
    weight_decay: float = 5e-2
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    clip_grad_norm: float = 1.0
    warmup_epochs: int = 100
    warmup_start_factor: float = 1e-10
    min_learning_rate: float = 1e-6

    # testing
    n_tests: int = 100
    test_episodes: int = 10
    max_episode_length: int = 5000


GYM_ENV_MAP = {
    "Alien": "AlienNoFrameskip-v4",
    "Assault": "AssaultNoFrameskip-v4",
    "Asterix": "AsterixNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "ChopperCommand": "ChopperCommandNoFrameskip-v4",
    "DemonAttack": "DemonAttackNoFrameskip-v4",
    "Enduro": "EnduroNoFrameskip-v4",
    "Freeway": "FreewayNoFrameskip-v4",
    "Frostbite": "FrostbiteNoFrameskip-v4",
    "MsPacman": "MsPacmanNoFrameskip-v4",
    "Phoenix": "PhoenixNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
    "RoadRunner": "RoadRunnerNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "UpNDown": "UpNDownNoFrameskip-v4",
}


def test_agent(
    args: Config, model: torch.nn.Module
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Runs the model in the actual Gym environment to measure performance.
    """
    env_name = GYM_ENV_MAP.get(args.game)
    if env_name is None:
        print(f"Warning: No Gym environment found for '{args.game}' in GYM_ENV_MAP.")
        return 0.0

    env = gym.make(env_name, render_mode="rgb_array")
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, 4)

    action_meanings = env.unwrapped.get_action_meanings()
    fire_a = -1
    if "FIRE" in action_meanings:
        fire_a = action_meanings.index("FIRE")

    total_return = 0
    best_return = -1
    best_rollout_obs = []
    best_rollout_g = []

    model.eval()
    for i in range(args.test_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        if fire_a != -1:
            obs, _, _, _, _ = env.step(fire_a)

        rollout_obs = []
        rollout_g = []
        while not done and steps < args.max_episode_length:
            steps += 1

            obs = torch.from_numpy(obs).float() / 255.0
            F, H, W = obs.shape
            obs = obs.view(1, F, 1, H, W).to(device=device)

            with torch.no_grad():
                pred_a, cls_attn = model(obs)

                action = pred_a.argmax(dim=1).item()

                cls_attn = cls_attn.mean(dim=2)
                cls_attn = cls_attn.view(
                    -1,
                    F,
                    H // args.spatial_patch_size[0],
                    W // args.spatial_patch_size[1],
                )  # (1, F, H / PH, W / PW)
                cls_attn = Fn.interpolate(
                    cls_attn,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )  # (1, F, H, W)
                cls_attn = cls_attn / cls_attn.sum(dim=(-1, -2), keepdim=True)
                cls_attn = cls_attn.squeeze().cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            color_obs = env.render()
            color_obs = cv2.resize(color_obs, (H, W))
            color_obs = color_obs.transpose(2, 0, 1)

            rollout_obs.append(color_obs)
            rollout_g.append(cls_attn[-1])

        total_return += ep_reward

        if ep_reward > best_return:
            best_return = ep_reward
            best_rollout_obs = rollout_obs
            best_rollout_g = rollout_g

    env.close()

    mean_return = total_return / args.test_episodes

    best_rollout_obs = np.stack(best_rollout_obs)

    best_rollout_g = np.stack(best_rollout_g)

    g_min = best_rollout_g.min()
    g_max = best_rollout_g.max()
    if g_max - g_min > 1e-8:
        best_rollout_g = (best_rollout_g - g_min) / (g_max - g_min)
    else:
        best_rollout_g = best_rollout_g

    cmap = plt.get_cmap("viridis")
    best_rollout_g = cmap(best_rollout_g)
    best_rollout_g = best_rollout_g[..., :3]
    best_rollout_g = (best_rollout_g * 255).astype(np.uint8)
    best_rollout_g = np.transpose(best_rollout_g, (0, 3, 1, 2))

    return mean_return, best_rollout_obs, best_rollout_g


def save_checkpoint(
    path: str,
    epoch: int,
    best_reward: float,
    wandb_id: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scheduler_state = None
    if scheduler is not None:
        scheduler_state = scheduler.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler.state_dict(),
        "best_reward": best_reward,
        "wandb_id": wandb_id,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
) -> Tuple[int, float, str | None]:
    if not os.path.exists(path):
        return 0, -float("inf"), None

    print(f"--> Found checkpoint! Resuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNGs if available
    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = [state.cpu() for state in rng["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    best_reward = checkpoint.get("best_reward", -float("inf"))
    wandb_id = checkpoint.get("wandb_id", None)

    print(f"--> Resumed at Epoch {start_epoch}, Best Reward: {best_reward:.4f}")
    return start_epoch, best_reward, wandb_id


def calculate_loss(
    args: Config,
    model: torch.nn.Module,
    class_weights: torch.Tensor,
    obs: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
):
    with autocast(device_type="cuda", dtype=torch.float16):
        pred_a, cls_attn = model(
            obs
        )  # pred_a: (B, n_actions), cls_attn: (B, F, SpatialHeads, T)

        # behavior cloning loss
        policy_loss = Fn.cross_entropy(pred_a, a, weight=class_weights)

        # gaze loss
        _, _, GH, GW = g.shape

        cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
        _, F, _ = cls_attn.shape
        cls_attn = cls_attn.view(
            -1, F, GH // args.spatial_patch_size[0], GW // args.spatial_patch_size[1]
        )  # (B, F, H / PH, W / PW)

        cls_attn = Fn.interpolate(
            cls_attn,
            size=(GH, GW),
            mode="bilinear",
            align_corners=False,
        )  # (B, F, H, W)

        # current_sum = cls_attn.sum(dim=(2, 3), keepdim=True) + 1e-8
        # cls_attn = cls_attn / current_sum
        #
        # gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
        # gaze_loss = gaze_loss.mean()

        B, F, H, W = cls_attn.shape
        cls_attn_flat = cls_attn.view(B * F, -1)
        gaze_flat = g.view(B * F, -1)

        eps = 1e-8
        cls_attn_flat = cls_attn_flat + eps
        gaze_flat = gaze_flat + eps

        cls_attn_flat = cls_attn_flat / cls_attn_flat.sum(dim=1, keepdim=True)
        gaze_flat = gaze_flat / gaze_flat.sum(dim=1, keepdim=True)

        gaze_loss = torch.sum(
            gaze_flat * (torch.log(gaze_flat) - torch.log(cls_attn_flat)), dim=1
        )
        gaze_loss = gaze_loss.mean()

        return pred_a, policy_loss, gaze_loss


def train(
    args: Config,
    observations: torch.Tensor,
    gaze_coords: torch.Tensor,
    actions: torch.Tensor,
):
    """
    Train a ViViT model.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_coords: (B, F, layers, 2)
    :param actions: (B)
    :return:
    """
    resume_path = f"{args.save_folder}/{args.game}/latest_checkpoint.pt"

    B, F, C, H, W = observations.shape
    n_actions = torch.max(actions).item() + 1

    all_actions = actions.view(-1).long()
    class_counts = torch.bincount(all_actions)
    num_classes = len(class_counts)

    safe_counts = class_counts.float()
    safe_counts[safe_counts == 0] = 1.0

    total_samples = len(all_actions)
    class_weights = total_samples / (num_classes * safe_counts)
    class_weights = torch.sqrt(class_weights)
    class_weights = torch.clamp(class_weights, min=1.0, max=10.0)
    class_weights = class_weights.to(device=device)

    model = AuxGazeFactorizedViViT(
        image_size=(H, W),
        patch_size=args.spatial_patch_size,
        frames=F,
        channels=C,
        n_classes=n_actions,
        dim=args.embedding_dim,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        spatial_heads=args.spatial_heads,
        temporal_heads=args.temporal_heads,
        dim_head=args.inner_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        use_flash_attn=True,
        return_cls_attn=True,
        use_temporal_mask=True,
    ).to(device=device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )

    decay_epochs = args.epochs - args.warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=args.min_learning_rate
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch, best_reward, wandb_id = load_checkpoint(
        resume_path, model, optimizer, scaler, scheduler
    )

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(
        entity="papaya147-ml",
        project="GABRIL-Atari-ViViT",
        config=args.__dict__,
        name=f"{args.algorithm}_GABRIL-Atari-{args.game}_bs={args.batch_size}_{date_str}",
        job_type="train",
        id=wandb_id,
        resume="allow",
    )

    dataset_len = len(observations)
    train_size = int(args.train_pct * dataset_len)
    val_size = dataset_len - train_size

    train_obs, val_obs = observations[:train_size], observations[train_size:]
    train_gaze, val_gaze = gaze_coords[:train_size], gaze_coords[train_size:]
    train_acts, val_acts = actions[:train_size], actions[train_size:]

    train_dataset = TensorDataset(train_obs, train_gaze, train_acts)
    val_dataset = TensorDataset(val_obs, val_gaze, val_acts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_interval = args.epochs / args.n_tests

    for e in range(start_epoch, args.epochs):
        metrics = {
            "train_loss": 0,
            "train_policy_loss": 0,
            "train_gaze_loss": 0,
            "train_acc": 0,
        }

        # train loop
        model.train()
        for obs, g, a in train_loader:
            obs, g = preprocess(args, obs, g)  # obs: (B, F, C, H, W), g: (B, F, H, W)
            a = a.to(device=device)  # (B, n_actions)

            optimizer.zero_grad()

            pred_a, policy_loss, gaze_loss = calculate_loss(
                args, model, class_weights, obs, g, a
            )  # pred_a: (B, n_actions)
            loss = policy_loss + args.lambda_gaze * gaze_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            acc = (pred_a.argmax(dim=1) == a).float().sum()

            curr_batch_size = obs.size(0)

            metrics["train_loss"] += loss.item() * curr_batch_size
            metrics["train_policy_loss"] += policy_loss.item() * curr_batch_size
            metrics["train_gaze_loss"] += gaze_loss.item() * curr_batch_size
            metrics["train_acc"] += acc.item()

        mean_reward = -1
        if (e + 1) % test_interval == 0:
            metrics["val_loss"] = 0
            metrics["val_policy_loss"] = 0
            metrics["val_gaze_loss"] = 0
            metrics["val_acc"] = 0

            # validation
            model.eval()
            with torch.no_grad():
                for obs, g, a in val_loader:
                    obs, g = preprocess(
                        args, obs, g, augment=False
                    )  # obs: (B, F, C, H, W), g: (B, F, H, W)
                    a = a.to(device=device)  # (B, n_actions)

                    pred_a, policy_loss, gaze_loss = calculate_loss(
                        args, model, class_weights, obs, g, a
                    )
                    loss = policy_loss + args.lambda_gaze * gaze_loss

                    acc = (pred_a.argmax(dim=1) == a).float().sum()

                    curr_batch_size = obs.size(0)

                    metrics["val_loss"] += loss.item() * curr_batch_size
                    metrics["val_policy_loss"] += policy_loss.item() * curr_batch_size
                    metrics["val_gaze_loss"] += gaze_loss.item() * curr_batch_size
                    metrics["val_acc"] += acc.item()

            # testing
            mean_reward, best_rollout_obs, best_rollout_g = test_agent(args, model)

            if mean_reward > best_reward:
                best_reward = mean_reward
                save_path = f"{args.save_folder}/{args.game}/best_reward.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        scheduler.step()

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e
        if mean_reward != -1:
            log_data["mean_reward"] = mean_reward
            log_data["best_rollout_obs"] = wandb.Video(
                best_rollout_obs, fps=4, format="gif"
            )
            log_data["best_rollout_g"] = wandb.Video(
                best_rollout_g, fps=4, format="gif"
            )
        log_data["learning_rate"] = optimizer.param_groups[0]["lr"]

        run.log(data=log_data)

        save_checkpoint(
            resume_path, e, best_reward, wandb_id, model, optimizer, scaler, scheduler
        )

    save_path = f"{args.save_folder}/{args.game}/final.pt"
    torch.save(model.state_dict(), save_path)

    run.finish()


def preprocess(
    args: Config,
    observations: torch.Tensor,
    gaze_coords: torch.Tensor,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the observations and gaze masks. # Convert the gaze masks into patches.
    Normalize the gaze patches.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_coords: (B, F, gaze_layers, 2)
    :param augment: Augment the data with random shifts, color jitter and noise?
    :return: (B, F, C, H, W), (B, F, H, W)
    """
    B, F, C, H, W = observations.shape
    random_example = random.randint(0, len(observations) - 1)

    gaze_masks = gaze.decaying_gaussian_mask(
        gaze_coords,
        shape=(H, W),
        base_sigma=args.gaze_sigma,
        temporal_decay=args.gaze_alpha,
        blur_growth=args.gaze_beta,
    )  # (B, F, H, W)

    # pre augmentation plots
    if args.use_plots:
        dataset.plot_frames(observations[random_example])
        dataset.plot_frames(gaze_masks.unsqueeze(2)[random_example])

    if augment:
        augment = Augment(
            frame_shape=(F, C, H, W),
            crop_padding=args.augment_crop_padding,
            light_intensity=args.augment_light_intensity,
            noise_std=args.augment_noise_std,
            p_pixel_dropout=args.augment_p_pixel_dropout,
            posterize_bits=args.augment_posterize_bits,
            blur_pixels=args.augment_blur_pixels,
            p_spatial_corruption=args.augment_p_spatial_corruptions,
            p_temporal_corruption=args.augment_p_temporal_corruptions,
        )
        observations, gaze_masks = augment(observations, gaze_masks)

    observations = observations.to(device=device)  # (B, F, C, H, W)
    gaze_masks = gaze_masks.to(device=device)  # (B, F, H, W)

    # # post augmentation plots
    if args.use_plots:
        dataset.plot_frames(observations[random_example])
        dataset.plot_frames(gaze_masks.unsqueeze(2)[random_example])

    # gaze_mask_patches = gaze.patchify(
    #     aug_gaze_masks, patch_size=args.spatial_patch_size
    # )
    # B, F, gridR, gridR, patchR, patchC = gaze_mask_patches.shape
    #
    # # plotting random gaze patches
    # if args.use_plots:
    #     gaze.plot_patches(gaze_mask_patches[random_example][0], 1)
    #
    # # pooling the last 2 dims of gaze
    # gaze_mask_patches = gaze_mask_patches.mean(dim=(-2, -1))
    # gaze_mask_patches = gaze_mask_patches.view(B, F, gridR * gridR)
    #
    # normalizing
    gaze_sums = gaze_masks.sum(dim=(-2, -1), keepdim=True)
    gaze_masks = gaze_masks / (gaze_sums + 1e-8)  # (B, F, H, W)

    return observations, gaze_masks  # gaze_mask_patches


def set_seed(seed: int):
    """
    Sets the seed for all sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = Config()

    set_seed(args.seed)

    if len(sys.argv) > 1:
        args.game_index = int(sys.argv[1])
    else:
        args.game_index = 0

    atari_games = dataset.list_games(args.atari_dataset_folder)

    args.game = atari_games[args.game_index]
    print(f"Game: {args.game}")

    observations, gaze_coords, actions = dataset.load_data(
        f"{args.atari_dataset_folder}/{args.game}",
        frame_stack=args.frame_stack,
        gaze_temporal_decay=args.gaze_alpha,
        device="cpu",
    )

    train(args, observations, gaze_coords, actions)


if __name__ == "__main__":
    main()
