import datetime
import os
import random
import sys
from typing import Literal, Tuple

import ale_py
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

import checkpoint
import dataset
import env_manager
from augmentation import Augment
from config import config
from device import device
from vivit import AuxGazeFactorizedViViT, FactorizedViViT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def evaluate_agent(model: torch.nn.Module, split: Literal["test", "val"]):
    """
    Runs the model in the actual Gym environment to measure performance.
    """
    start_fire = config.game in ["Breakout"]

    env = env_manager.create_env(
        env_name=config.game,
        noop_max=0,
        frame_skip=config.frame_skip,
        obs_size=84,
        action_repeat_probability=0.25,
        num_stack=config.frame_stack,
        start_fire=start_fire,
    )

    ep_returns = []
    ep_steps = []
    best_return = -1
    best_rollout_obs = []
    best_rollout_g = []
    best_rollout_overlaid = []

    episodes = config.test_episodes if split == "test" else config.val_episodes

    model.eval()
    for i in range(episodes):
        seed = config.seed + i
        obs, _ = env.reset(seed=seed)
        done = False
        ep_return = 0
        steps = 0

        rollout_obs = []
        rollout_g = []
        while not done and steps < config.max_episode_length:
            steps += 1

            obs = torch.from_numpy(np.array(obs)).float() / 255.0
            F, H, W = obs.shape
            obs = obs.view(1, F, 1, H, W).to(device=device)

            color_obs = env.render()
            color_obs = cv2.resize(color_obs, (H, W))
            color_obs = color_obs.transpose(2, 0, 1)

            with torch.no_grad():
                pred_a, cls_attn = model(obs)

                action = pred_a.argmax(dim=1).item()

                cls_attn = cls_attn.mean(dim=2)
                cls_attn = cls_attn.view(
                    -1,
                    F,
                    H // config.spatial_patch_size[0],
                    W // config.spatial_patch_size[1],
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
            ep_return += reward

            if split == "val":
                rollout_obs.append(color_obs)
                rollout_g.append(cls_attn[-1])

        ep_returns.append(ep_return)
        ep_steps.append(steps)

        if split == "val" and ep_return > best_return:
            best_return = ep_return
            best_rollout_obs = rollout_obs
            best_rollout_g = rollout_g

    env.close()

    ep_returns = np.array(ep_returns)

    ep_steps = np.array(ep_steps)

    if split == "val":
        best_rollout_obs = np.stack(best_rollout_obs)

        best_rollout_g = np.stack(best_rollout_g)

        g_min = best_rollout_g.min(axis=(1, 2), keepdims=True)
        g_max = best_rollout_g.max(axis=(1, 2), keepdims=True)
        denominator = g_max - g_min
        denominator[denominator < 1e-8] = 1.0
        best_rollout_g = (best_rollout_g - g_min) / denominator

        best_rollout_overlaid = best_rollout_obs * np.expand_dims(
            best_rollout_g, axis=1
        )

        best_rollout_g = np.power(
            best_rollout_g, 0.5
        )  # making colors brighter, optional
        cmap = plt.get_cmap("viridis")
        best_rollout_g = cmap(best_rollout_g)
        best_rollout_g = best_rollout_g[..., :3]  # getting rid of alpha channel
        best_rollout_g = (best_rollout_g * 255).astype(np.uint8)
        best_rollout_g = np.transpose(best_rollout_g, (0, 3, 1, 2))

    return (
        ep_returns,
        ep_steps,
        best_rollout_obs,
        best_rollout_g,
        best_rollout_overlaid,
    )


def calculate_loss(
    model: torch.nn.Module,
    class_weights: torch.Tensor,
    obs: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
):
    with autocast(device_type=device, dtype=torch.float16):
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
            -1,
            F,
            GH // config.spatial_patch_size[0],
            GW // config.spatial_patch_size[1],
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
    observations: torch.Tensor,
    gaze_masks: torch.Tensor,
    actions: torch.Tensor,
):
    """
    Train a ViViT model.

    :param observations: (B, F, C, H, W)
    :param gaze_masks: (B, F, H, W)
    :param actions: (B)
    :return:
    """
    resume_path = f"{config.save_folder}/{config.game}/latest_checkpoint.pt"

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

    if config.algorithm == "AuxGazeFactorizedViViT":
        model = AuxGazeFactorizedViViT(
            image_size=(H, W),
            patch_size=config.spatial_patch_size,
            frames=F,
            channels=C,
            n_classes=n_actions,
            dim=config.embedding_dim,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            spatial_heads=config.spatial_heads,
            temporal_heads=config.temporal_heads,
            dim_head=config.inner_dim,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            use_flash_attn=True,
            return_cls_attn=True,
            use_temporal_mask=True,
        )
    else:
        model = FactorizedViViT(
            image_size=(H, W),
            patch_size=config.spatial_patch_size,
            frames=F,
            channels=C,
            n_classes=n_actions,
            dim=config.embedding_dim,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            spatial_heads=config.spatial_heads,
            temporal_heads=config.temporal_heads,
            dim_head=config.inner_dim,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            use_flash_attn=True,
            return_cls_attn=True,
            use_temporal_mask=True,
        )
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = GradScaler()

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )

    decay_epochs = config.epochs - config.warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=config.min_learning_rate
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    start_epoch, best_return, wandb_id = checkpoint.load(
        resume_path, model, optimizer, scaler, scheduler
    )

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(
        entity="papaya147-ml",
        project="FactorizedViViT-GABRIL-Atari",
        config=config.__dict__,
        name=f"{config.algorithm}_GABRIL-Atari-{config.game}_bs={config.batch_size}_{date_str}",
        job_type="train",
        id=wandb_id,
        resume="allow",
    )

    dataset_len = len(observations)
    train_size = int(config.train_pct * dataset_len)
    val_size = dataset_len - train_size

    train_obs, val_obs = observations[:train_size], observations[train_size:]
    train_gaze, val_gaze = (
        gaze_masks[:train_size],
        gaze_masks[train_size:],
    )
    train_acts, val_acts = actions[:train_size], actions[train_size:]

    train_dataset = TensorDataset(train_obs, train_gaze, train_acts)
    val_dataset = TensorDataset(val_obs, val_gaze, val_acts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    for e in range(start_epoch, config.epochs):
        metrics = {
            "train/train_loss": 0,
            "train/train_policy_loss": 0,
            "train/train_gaze_loss": 0,
            "train/train_acc": 0,
        }

        # train loop
        model.train()
        for obs, g, a in train_loader:
            obs, g = preprocess(obs, g)  # obs: (B, F, C, H, W), g: (B, F, H, W)
            a = a.to(device=device)  # (B, n_actions)

            optimizer.zero_grad()

            pred_a, policy_loss, gaze_loss = calculate_loss(
                model, class_weights, obs, g, a
            )  # pred_a: (B, n_actions)
            loss = policy_loss + config.lambda_gaze * gaze_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            acc = (pred_a.argmax(dim=1) == a).float().sum()

            curr_batch_size = obs.size(0)

            metrics["train/train_loss"] += loss.item() * curr_batch_size
            metrics["train/train_policy_loss"] += policy_loss.item() * curr_batch_size
            metrics["train/train_gaze_loss"] += gaze_loss.item() * curr_batch_size
            metrics["train/train_acc"] += acc.item()

        mean_return = -1
        if (e + 1) % config.val_interval == 0:
            metrics["eval/val_loss"] = 0
            metrics["eval/val_policy_loss"] = 0
            metrics["eval/val_gaze_loss"] = 0
            metrics["eval/val_acc"] = 0

            # validation
            model.eval()
            with torch.no_grad():
                for obs, g, a in val_loader:
                    obs, g = preprocess(
                        obs, g, augment=False
                    )  # obs: (B, F, C, H, W), g: (B, F, H, W)
                    a = a.to(device=device)  # (B, n_actions)

                    pred_a, policy_loss, gaze_loss = calculate_loss(
                        model, class_weights, obs, g, a
                    )
                    loss = policy_loss + config.lambda_gaze * gaze_loss

                    acc = (pred_a.argmax(dim=1) == a).float().sum()

                    curr_batch_size = obs.size(0)

                    metrics["eval/val_loss"] += loss.item() * curr_batch_size
                    metrics["eval/val_policy_loss"] += (
                        policy_loss.item() * curr_batch_size
                    )
                    metrics["eval/val_gaze_loss"] += gaze_loss.item() * curr_batch_size
                    metrics["eval/val_acc"] += acc.item()

            # testing
            (
                ep_returns,
                ep_steps,
                best_rollout_obs,
                best_rollout_g,
                best_rollout_overlaid,
            ) = evaluate_agent(model=model, split="val")
            mean_return = float(ep_returns.mean())

            if mean_return > best_return:
                best_return = mean_return
                final_save_path = f"{config.save_folder}/{config.game}/best_return.pt"
                os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
                torch.save(model.state_dict(), final_save_path)

        scheduler.step()

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e
        log_data["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        if mean_return != -1:
            log_data["eval/mean_return"] = mean_return
            log_data["eval/std_return"] = float(ep_returns.std())
            log_data["eval/max_return"] = float(ep_returns.max())
            log_data["eval/min_return"] = float(ep_returns.min())

            log_data["eval/mean_steps"] = float(ep_steps.mean())
            log_data["eval/std_steps"] = float(ep_steps.std())
            log_data["eval/best_rollout_obs"] = wandb.Video(
                best_rollout_obs, fps=15, format="gif"
            )
            log_data["eval/best_rollout_g"] = wandb.Video(
                best_rollout_g, fps=15, format="gif"
            )
            log_data["eval/best_rollout_overlaid"] = wandb.Video(
                best_rollout_overlaid, fps=15, format="gif"
            )

        run.log(data=log_data)

        checkpoint.save(
            resume_path, e, best_return, wandb_id, model, optimizer, scaler, scheduler
        )

    final_save_path = os.path.join(config.save_folder, config.game, "final.pt")
    torch.save(model.state_dict(), final_save_path)

    best_save_path = os.path.join(config.save_folder, config.game, "best_return.pt")
    if config.test_model == "best":
        model.load_state_dict(torch.load(best_save_path))

    ep_returns, ep_steps, _, _, _ = evaluate_agent(model=model, split="test")
    run.summary["test/mean_return"] = np.mean(ep_returns)
    run.summary["test/std_return"] = np.std(ep_returns)
    run.summary["test/max_return"] = np.max(ep_returns)
    run.summary["test/min_return"] = np.min(ep_returns)

    table = wandb.Table(data=[[r] for r in ep_returns], columns=["return"])
    run.log({"test/return_distribution": wandb.plot.histogram(table, "return")})

    final_model = wandb.Artifact(f"{run.name}-final-model", type="model")
    final_model.add_file(final_save_path)
    run.log_artifact(final_model)

    best_model = wandb.Artifact(f"{run.name}-best-model", type="model")
    best_model.add_file(best_save_path)
    run.log_artifact(best_model)

    run.finish()


def preprocess(
    observations: torch.Tensor,
    gaze_masks: torch.Tensor,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the observations and gaze masks. Normalize the gaze masks.

    :param observations: (B, F, C, H, W)
    :param gaze_masks: (B, F, H, W)
    :param augment: Augment the data with random shifts, color jitter and noise?
    :return: (B, F, C, H, W), (B, F, H, W)
    """
    B, F, C, H, W = observations.shape
    random_example = random.randint(0, len(observations) - 1)

    # pre augmentation plots
    if config.use_plots:
        dataset.plot_frames(observations[random_example])
        dataset.plot_frames(gaze_masks.unsqueeze(2)[random_example])

    if augment:
        augment = Augment(
            frame_shape=(F, C, H, W),
            crop_padding=config.augment_crop_padding,
            cutout_hole_size=config.augment_cutout_hole_size,
            p_spatial_corruption=config.augment_p_spatial_corruptions,
        )
        observations, gaze_masks = augment(observations, gaze_masks)

    observations = observations.to(device=device)  # (B, F, C, H, W)
    gaze_masks = gaze_masks.to(device=device)  # (B, F, H, W)

    # # post augmentation plots
    if config.use_plots:
        dataset.plot_frames(observations[random_example])
        dataset.plot_frames(gaze_masks.unsqueeze(2)[random_example])

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
    set_seed(config.seed)

    print(f"Game: {config.game}")

    if config.loading_method == "mine":
        observations, gaze_masks, actions = dataset.load_data()
    elif config.loading_method == "gabril":
        observations, actions, gaze_masks, _ = dataset.gabril_load_data(
            num_episodes=dataset.MAX_EPISODES[config.game],
        )
    else:
        raise ValueError(f"Unknown loading method: {config.loading_method}")

    train(observations, gaze_masks, actions)


if __name__ == "__main__":
    main()
