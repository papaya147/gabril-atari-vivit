import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import ale_py
import gymnasium as gym
import torch
import torch.nn.functional as Fn
import torch.optim as optim
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from torch.utils.data import DataLoader, TensorDataset, random_split

import atari
import augmentation
import gaze
import wandb
from device import device
from vivit import FactorizedViViT

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

    # gaze
    gaze_sigma: int = 5
    gaze_beta: float = 0.99
    gaze_alpha: float = 0.7

    # augmentation
    augment_shift_pad: int = 4
    augment_noise_std: float = 0.005

    # transformer arch
    spatial_patch_size: Tuple[int, int] = (6, 6)
    embedding_dim: int = 256
    spatial_depth: int = 3
    temporal_depth: int = 2
    spatial_heads: int = 8
    temporal_heads: int = 8
    inner_dim: int = 64
    mlp_dim: int = 256
    dropout: float = 0.1

    # hyperparams
    learning_rate: float = 1e-4
    epochs: int = 1000
    train_pct: float = 0.8
    batch_size: int = 32
    lambda_gaze: float = 10.0
    weight_decay: float = 0.01
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # testing
    test_episodes: int = 10
    max_episode_length: int = 10000


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


def test_agent(args: Config, model: torch.nn.Module) -> float:
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

    total_reward = 0

    model.eval()
    for i in range(args.test_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        if fire_a != -1:
            obs, _, _, _, _ = env.step(fire_a)

        while not done and steps < args.max_episode_length:
            steps += 1

            obs = torch.from_numpy(obs).float() / 255.0
            F, H, W = obs.shape
            obs = obs.view(1, F, 1, H, W).to(device=device)

            with torch.no_grad():
                pred_a, _ = model(obs)
                action = pred_a.argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_reward += ep_reward

    env.close()

    return total_reward / args.test_episodes


def save_checkpoint(
    path: str,
    epoch: int,
    best_reward: float,
    wandb_id: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = None,
):
    """
    Saves the complete training state to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scheduler_state = None
    if scheduler is not None:
        scheduler_state = scheduler.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "best_reward": best_reward,
        "wandb_id": wandb_id,
    }
    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = None,
) -> Tuple[int, float, str | None]:
    """
    Attempts to load a checkpoint.
    Returns: (start_epoch, best_reward, wandb_id)
    If no checkpoint is found, returns defaults: (0, -inf, None)
    """
    if not os.path.exists(path):
        return 0, -float("inf"), None

    print(f"--> Found checkpoint! Resuming from {path}")
    checkpoint = torch.load(path, map_location=device)

    # Load states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    best_reward = checkpoint.get("best_reward", -float("inf"))
    wandb_id = checkpoint.get("wandb_id", None)

    print(f"--> Resumed at Epoch {start_epoch}, Best Reward: {best_reward:.4f}")
    return start_epoch, best_reward, wandb_id


def train(
    args: Config,
    observations: torch.Tensor,
    gaze_masks: torch.Tensor,
    actions: torch.Tensor,
):
    """
    Train a ViViT model.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_masks: (B, F, H, W)
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
    class_weights[class_counts == 0] = 0.0

    class_weights = class_weights.to(device=device)

    model = FactorizedViViT(
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
        use_flash_attn=False,
        return_cls_attn=True,
        use_temporal_mask=True,
    ).to(device=device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    start_epoch, best_reward, wandb_id = load_checkpoint(
        resume_path, model, optimizer, scheduler=None
    )

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    group_id = (
        f"v1_"
        f"lr{args.learning_rate:.0e}_"
        f"lam{args.lambda_gaze}_"
        f"dim{args.embedding_dim}_"
        f"pt{args.spatial_patch_size[0]}_"
        f"d{args.dropout}"
    )
    run = wandb.init(
        entity="papaya147-ml",
        project="ViViT-Atari",
        config=args.__dict__,
        group=group_id,
        name=f"{args.game}-v3",
        job_type="train",
        id=wandb_id,
        resume="allow",
    )

    dataset = TensorDataset(observations, gaze_masks, actions)
    train_size = int(args.train_pct * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    for e in range(start_epoch, args.epochs):
        metrics = {
            "train_loss": 0,
            "train_policy_loss": 0,
            "train_gaze_loss": 0,
            "train_acc": 0,
            "val_loss": 0,
            "val_policy_loss": 0,
            "val_gaze_loss": 0,
            "val_acc": 0,
        }

        # train loop
        model.train()
        for obs, g, a in train_loader:
            obs, g = preprocess(args, obs, g)
            a = a.to(device=device)
            pred_a, cls_attn = model(obs)

            # behavior cloning loss
            policy_loss = Fn.cross_entropy(pred_a, a, weight=class_weights)

            # gaze loss
            cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
            gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
            gaze_loss = gaze_loss.mean()

            loss = policy_loss + args.lambda_gaze * gaze_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (pred_a.argmax(dim=1) == a).float().sum()

            curr_batch_size = obs.size(0)

            metrics["train_loss"] += loss.item() * curr_batch_size
            metrics["train_policy_loss"] += policy_loss.item() * curr_batch_size
            metrics["train_gaze_loss"] += gaze_loss.item() * curr_batch_size
            metrics["train_acc"] += acc.item()

        # validation
        model.eval()
        with torch.no_grad():
            for obs, g, a in val_loader:
                obs, g = preprocess(args, obs, g, augment=False)
                a = a.to(device=device)
                pred_a, cls_attn = model(obs)

                # behavior cloning loss
                policy_loss = Fn.cross_entropy(pred_a, a, weight=class_weights)

                # gaze loss
                cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
                gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
                gaze_loss = gaze_loss.mean()

                loss = policy_loss + args.lambda_gaze * gaze_loss

                acc = (pred_a.argmax(dim=1) == a).float().sum()

                curr_batch_size = obs.size(0)

                metrics["val_loss"] += loss.item() * curr_batch_size
                metrics["val_policy_loss"] += policy_loss.item() * curr_batch_size
                metrics["val_gaze_loss"] += gaze_loss.item() * curr_batch_size
                metrics["val_acc"] += acc.item()

        scheduler.step(metrics["val_loss"])

        # testing
        mean_reward = test_agent(args, model)

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e
        log_data["reward"] = mean_reward

        run.log(data=log_data)

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = f"{args.save_folder}/{args.game}/best_reward.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        save_checkpoint(
            resume_path, e, best_reward, wandb_id, model, optimizer, scheduler=None
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
    Augment the observations and gaze masks. Convert the gaze masks into patches.
    Normalize the gaze patches.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_coords: (B, F, layers, 2)
    :param augment: Augment the data with random shifts and noise?
    :return: (B, F, C, H, W), (B, F, T)
    """
    B, F, C, H, W = observations.shape
    random_example = random.randint(0, len(observations) - 1)

    gaze_masks = gaze.decaying_gaussian_mask(
        gaze_coords,
        sigma=args.gaze_sigma,
        shape=(H, W),
        beta=args.gaze_beta,
        alpha=args.gaze_alpha,
    )

    aug_observations = observations.to(device=device)
    aug_gaze_masks = gaze_masks.to(device=device)
    if augment:
        aug_observations, aug_gaze_masks = augmentation.random_shift(
            aug_observations, aug_gaze_masks, pad=args.augment_shift_pad
        )
        aug_observations, aug_gaze_masks = augmentation.random_noise(
            aug_observations, aug_gaze_masks, std=args.augment_noise_std
        )

    # plotting random observations and gazes
    if args.use_plots:
        atari.plot_frames(aug_observations[random_example])
        atari.plot_frames(aug_gaze_masks.unsqueeze(2)[random_example])

    gaze_mask_patches = gaze.patchify(
        aug_gaze_masks, patch_size=args.spatial_patch_size
    )
    B, F, gridR, gridR, patchR, patchC = gaze_mask_patches.shape

    # plotting random gaze patches
    if args.use_plots:
        gaze.plot_patches(gaze_mask_patches[random_example][0], 1)

    # pooling the last 2 dims of gaze
    gaze_mask_patches = gaze_mask_patches.mean(dim=(-2, -1))
    gaze_mask_patches = gaze_mask_patches.view(B, F, gridR * gridR)

    # normalizing
    gaze_sums = gaze_mask_patches.sum(dim=-1, keepdim=True)
    gaze_mask_patches = gaze_mask_patches / (gaze_sums + 1e-8)

    return aug_observations, gaze_mask_patches


def main():
    args = Config()

    if len(sys.argv) > 1:
        args.game_index = int(sys.argv[1])
    else:
        args.game_index = 0

    atari_games = atari.list_games(args.atari_dataset_folder)

    args.game = atari_games[args.game_index]
    print(f"Game: {args.game}")

    observations, gaze_coords, actions = atari.load_data(
        f"{args.atari_dataset_folder}/{args.game}",
        device="cpu",
    )

    train(args, observations, gaze_coords, actions)


if __name__ == "__main__":
    main()
