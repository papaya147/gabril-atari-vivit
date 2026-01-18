import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as Fn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, TensorDataset, random_split

import atari
import augmentation
import gaze
import wandb
from device import device
from vivit import FactorizedViViT


@dataclass
class Config:
    # paths and flags
    game: str = ""
    atari_dataset_folder: str = "./atari-dataset"
    use_plots: bool = False
    save_folder: str = "./models"

    # gaze
    gaze_sigma: int = 5
    gaze_decay: float = 0.9

    # augmentation
    augment_shift_pad: int = 4
    augment_noise_std: float = 0.005

    # transformer arch
    spatial_patch_size: Tuple[int, int] = (6, 6)
    embedding_dim: int = 256
    spatial_depth: int = 4
    temporal_depth: int = 4
    spatial_heads: int = 8
    temporal_heads: int = 8
    inner_dim: int = 64
    mlp_dim: int = 256
    dropout: float = 0.0

    # hyperparams
    learning_rate: float = 1e-4
    epochs: int = 1000
    train_pct: float = 0.8
    batch_size: int = 32
    lambda_gaze: float = 1.0


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
    run = wandb.init(
        entity="papaya147-ml",
        project="ViViT-Atari",
        config=args.__dict__,
    )

    B, F, C, H, W = observations.shape
    n_actions = torch.max(actions).item() + 1

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
    ).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    dataset = TensorDataset(observations, gaze_masks, actions)
    train_size = int(args.train_pct * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    for e in range(args.epochs):
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
            pred_a, cls_attn = model(obs)

            # behavior cloning loss
            policy_loss = Fn.cross_entropy(pred_a, a)

            # gaze loss
            cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
            gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
            gaze_loss = gaze_loss.mean()

            loss = policy_loss + args.lambda_gaze * gaze_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (pred_a.argmax(dim=1) == a).float().sum()

            metrics["train_loss"] += loss.item()
            metrics["train_policy_loss"] += policy_loss.item()
            metrics["train_gaze_loss"] += gaze_loss.item()
            metrics["train_acc"] += acc.item()

        # validation
        model.eval()
        with torch.no_grad():
            for obs, g, a in val_loader:
                obs, g = preprocess(args, obs, g, augment=False)
                pred_a, cls_attn = model(obs)

                # behavior cloning loss
                policy_loss = Fn.cross_entropy(pred_a, a)

                # gaze loss
                cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
                gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
                gaze_loss = gaze_loss.mean()

                loss = policy_loss + args.lambda_gaze * gaze_loss

                acc = (pred_a.argmax(dim=1) == a).float().sum()

                metrics["val_loss"] += loss.item()
                metrics["val_policy_loss"] += policy_loss.item()
                metrics["val_gaze_loss"] += gaze_loss.item()
                metrics["val_acc"] += acc.item()

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e

        run.log(data=log_data)

        # checkpointing
        if e % 10 == 0:
            save_path = f"{args.save_folder}/{args.game}/{e}.pt"
            print(f"Saving model {save_path}...")
            torch.save(model.state_dict(), save_path)

    save_path = f"{args.save_folder}/{args.game}/final.pt"
    torch.save(model.state_dict(), save_path)


def preprocess(
    args: Config,
    observations: torch.Tensor,
    gaze_masks: torch.Tensor,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the observations and gaze masks. Convert the gaze masks into patches.
    Normalize the gaze patches.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_masks: (B, F, H, W)
    :param augment: Augment the data with random shifts and noise?
    :return: (B, F, C, H, W), (B, F, T)
    """
    B, F, C, H, W = observations.shape
    random_example = random.randint(0, len(observations) - 1)

    aug_observations = observations
    aug_gaze_masks = gaze_masks
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

    atari_games = atari.list_games(args.atari_dataset_folder)

    for game in atari_games:
        args.game = game

        observations, gaze_coords, actions = atari.load_data(
            f"{args.atari_dataset_folder}/{game}/num_episodes_20_fs4_human.pt",
            device="mps",
        )
        B, F, C, H, W = observations.shape

        gaze_masks = gaze.decaying_gaussian_mask(
            gaze_coords, sigma=args.gaze_sigma, shape=(H, W), decay=args.gaze_decay
        )

        train(args, observations, gaze_masks, actions)
        break


if __name__ == "__main__":
    main()
