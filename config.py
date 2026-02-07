import argparse
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # paths and flags
    game: str
    atari_dataset_folder: str
    loading_method: str
    use_plots: bool
    save_folder: str
    seed: int
    algorithm: str

    frame_stack: int
    frame_skip: int

    # gaze
    gaze_sigma: float
    gaze_beta: float
    gaze_alpha: float

    # augmentation
    augment_crop_padding: int
    augment_cutout_hole_size: int
    augment_light_intensity: float
    augment_noise_std: float
    augment_p_pixel_dropout: float
    augment_posterize_bits: int
    augment_blur_pixels: int
    augment_p_spatial_corruptions: float
    augment_p_temporal_corruptions: float

    # transformer
    spatial_patch_size: Tuple[int, int]
    embedding_dim: int
    spatial_depth: int
    temporal_depth: int
    spatial_heads: int
    temporal_heads: int
    inner_dim: int
    mlp_dim: int
    dropout: float

    # hyperparams
    learning_rate: float
    epochs: int
    train_pct: float
    batch_size: int
    lambda_gaze: float
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int
    clip_grad_norm: float
    warmup_epochs: int
    warmup_start_factor: float
    min_learning_rate: float

    # validation per epoch
    val_interval: int
    val_episodes: int
    max_episode_length: int

    # testing
    test_model: str
    test_episodes: int


parser = argparse.ArgumentParser()

parser.add_argument("--game", type=str, default="Alien")
parser.add_argument("--atari-dataset-folder", type=str, default="./atari-dataset")
parser.add_argument(
    "--loading-method",
    type=str,
    choices=["mine", "gabril"],
    default="mine",
    help="Data loading method: 'mine' or 'gabril'",
)
parser.add_argument("--use-plots", action="store_true", default=False)
parser.add_argument("--save-folder", type=str, default="./models")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["FactorizedViViT", "AuxGazeFactorizedViViT"],
    default="AuxGazeFactorizedViViT",
)

# frame handling
parser.add_argument("--frame-stack", type=int, default=4)
parser.add_argument("--frame-skip", type=int, default=4)

# gaze parameters
parser.add_argument("--gaze-sigma", type=int, default=15)
parser.add_argument("--gaze-beta", type=float, default=0.99)
parser.add_argument("--gaze-alpha", type=float, default=0.7)

# augmentation parameters
parser.add_argument("--aug-crop-padding", type=int, default=4)
parser.add_argument("--aug-cutout-hole-size", type=int, default=12)
parser.add_argument("--aug-light-intensity", type=float, default=0.2)
parser.add_argument("--aug-noise-std", type=float, default=0.01)
parser.add_argument("--aug-p-pixel-dropout", type=float, default=0.01)
parser.add_argument("--aug-posterize-bits", type=int, default=4)
parser.add_argument("--aug-blur-pixels", type=int, default=2)
parser.add_argument("--aug-p-spatial", type=float, default=0.5)
parser.add_argument("--aug-p-temporal", type=float, default=0.25)

# transformer arch
parser.add_argument(
    "--patch-size", type=int, default=6, help="Spatial patch size (square)"
)
parser.add_argument("--emb-dim", type=int, default=256)
parser.add_argument("--spatial-depth", type=int, default=4)
parser.add_argument("--temporal-depth", type=int, default=2)
parser.add_argument("--spatial-heads", type=int, default=4)
parser.add_argument("--temporal-heads", type=int, default=4)
parser.add_argument("--inner-dim", type=int, default=64)
parser.add_argument("--mlp-dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)

# hyperparams
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument(
    "--train-pct", type=float, default=0.95, help="Percentage of data for training"
)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument(
    "--lambda-gaze", type=float, default=0.1, help="Weight for gaze auxiliary loss"
)
parser.add_argument("--weight-decay", type=float, default=5e-2)
parser.add_argument("--scheduler-factor", type=float, default=0.5)
parser.add_argument("--scheduler-patience", type=int, default=5)
parser.add_argument("--clip-grad-norm", type=float, default=1.0)
parser.add_argument("--warmup-epochs", type=int, default=20)
parser.add_argument("--warmup-start-factor", type=float, default=1e-10)
parser.add_argument("--min-lr", type=float, default=1e-6)

# validation per epoch
parser.add_argument("--val-interval", type=int, default=10)
parser.add_argument("--val-episodes", type=int, default=10)
parser.add_argument("--max-episode-length", type=int, default=5000)

# testing
parser.add_argument(
    "--test-model",
    type=str,
    choices=["best", "final"],
    default="best",
    help="Which saved model to use for testing: 'best' or 'final'",
)
parser.add_argument("--test-episodes", type=int, default=100)

args = parser.parse_args()

config = Config(
    # paths and flags
    game=args.game,
    atari_dataset_folder=args.atari_dataset_folder,
    loading_method=args.loading_method,
    use_plots=args.use_plots,
    save_folder=args.save_folder,
    seed=args.seed,
    algorithm=args.algorithm,
    frame_stack=args.frame_stack,
    frame_skip=args.frame_skip,
    # gaze
    gaze_sigma=args.gaze_sigma,
    gaze_beta=args.gaze_beta,
    gaze_alpha=args.gaze_alpha,
    # augmentation
    augment_crop_padding=args.aug_crop_padding,
    augment_cutout_hole_size=args.aug_cutout_hole_size,
    augment_light_intensity=args.aug_light_intensity,
    augment_noise_std=args.aug_noise_std,
    augment_p_pixel_dropout=args.aug_p_pixel_dropout,
    augment_posterize_bits=args.aug_posterize_bits,
    augment_blur_pixels=args.aug_blur_pixels,
    augment_p_spatial_corruptions=args.aug_p_spatial,
    augment_p_temporal_corruptions=args.aug_p_temporal,
    # transformer arch
    spatial_patch_size=(args.patch_size, args.patch_size),
    embedding_dim=args.emb_dim,
    spatial_depth=args.spatial_depth,
    temporal_depth=args.temporal_depth,
    spatial_heads=args.spatial_heads,
    temporal_heads=args.temporal_heads,
    inner_dim=args.inner_dim,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    # hyperparams
    learning_rate=args.lr,
    epochs=args.epochs,
    train_pct=args.train_pct,
    batch_size=args.batch_size,
    lambda_gaze=args.lambda_gaze,
    weight_decay=args.weight_decay,
    scheduler_factor=args.scheduler_factor,
    scheduler_patience=args.scheduler_patience,
    clip_grad_norm=args.clip_grad_norm,
    warmup_epochs=args.warmup_epochs,
    warmup_start_factor=args.warmup_start_factor,
    min_learning_rate=args.min_lr,
    # validation per epoch
    val_interval=args.val_interval,
    val_episodes=args.val_episodes,
    max_episode_length=args.max_episode_length,
    # testing
    test_model=args.test_model,
    test_episodes=args.test_episodes,
)
