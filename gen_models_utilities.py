from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

plt.style.use("seaborn-v0_8-muted")
sns.set_context("talk")

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# Set device priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(RNG_SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _compute_axis_limits(
    points: Optional[np.ndarray], pad: float = 0.15
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if points is None:
        return (-1.0, 1.0), (-1.0, 1.0)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    if pts.ndim == 1:
        if pts.shape[0] == 2:
            pts = pts.reshape(1, 2)
        else:
            raise ValueError(
                "Expected points with shape (N, 2) or (2,) for a single point"
            )
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_span = 0.5 * (maxs - mins)
    radius = float(np.max(half_span))
    if radius <= 0:
        radius = 1.0
    radius *= 1 + pad
    return (center[0] - radius, center[0] + radius), (
        center[1] - radius,
        center[1] + radius,
    )


def create_ring_gaussians(
    n_samples: int = 5000,
    n_modes: int = 8,
    radius: float = 3.0,
    cluster_std: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    assignments = np.random.choice(n_modes, size=n_samples)
    noise = np.random.randn(n_samples, 2) * cluster_std
    points = centers[assignments] + noise
    return points.astype(np.float32), assignments.astype(np.int64)


def make_loader(data: np.ndarray, batch_size: int) -> DataLoader:
    tensor = torch.from_numpy(data).to(torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def compute_diversity_metric(samples: torch.Tensor) -> float:
    data = samples.detach().cpu().numpy()
    return float(np.var(data, axis=0).mean())


@dataclass
class GanHistory:
    d_loss: List[float]
    g_loss: List[float]
    diversity: List[float]
    real_scores: List[float]
    fake_scores: List[float]
    # snapshots will store checkpoint information captured during training
    snapshots: Optional[List[Dict]] = None

    def __post_init__(self) -> None:
        if self.snapshots is None:
            self.snapshots = []

    def append_epoch(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            getattr(self, key).append(value)


@dataclass
class SinkhornHistory:
    loss: List[float]
    diversity: List[float]


def plot_model_diagnostics(
    history,
    real_samples: np.ndarray,
    generator: nn.Module,
    noise_dim: int = 2,
    title_suffix: str = "",
) -> None:
    is_adversarial = hasattr(history, "d_loss") and hasattr(history, "g_loss")
    if is_adversarial:
        epochs = np.arange(1, len(history.d_loss) + 1)
    else:
        epochs = np.arange(1, len(history.loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    if is_adversarial:
        ax.plot(epochs, history.d_loss, label="Discriminator", linewidth=2)
        ax.plot(epochs, history.g_loss, label="Generator", linewidth=2)
    else:
        ax.plot(epochs, history.loss, label="Loss", color="tab:blue", linewidth=2)

    ax.set_title(f"Training Losses {title_suffix}")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    # Set the axis to log scale
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    if hasattr(history, "diversity") and len(getattr(history, "diversity", [])) > 0:
        ax.plot(epochs, history.diversity, color="teal", linewidth=2)
        ax.set_title("Sample Diversity Over Time")
        ax.set_ylabel("Variance score")
    else:
        ax.text(0.5, 0.5, "Diversity not available", ha="center", va="center")
        ax.set_title("Diversity")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(2000, noise_dim, device=device)
        samples = generator(noise).cpu().numpy()
    generator.train()
    ax.scatter(
        real_samples[:, 0],
        real_samples[:, 1],
        c="lightgray",
        s=8,
        alpha=0.3,
        label="Real",
    )
    ax.scatter(
        samples[:, 0], samples[:, 1], c="tomato", s=10, alpha=0.5, label="Generated"
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Generated vs. Real Samples")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    ax = axes[1, 1]
    if is_adversarial:
        if hasattr(history, "real_scores") and hasattr(history, "fake_scores"):
            ax.plot(epochs, history.real_scores, label="Real score", linewidth=2)
            ax.plot(epochs, history.fake_scores, label="Fake score", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title("Critic Outputs Over Time")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No critic scores", ha="center", va="center")
            ax.set_title("Critic Outputs")
    else:
        if len(history.loss) > 0:
            loss_array = np.asarray(history.loss, dtype=float)
            window = max(1, len(loss_array) // 20)
            if window > 1:
                smooth = np.convolve(loss_array, np.ones(window) / window, mode="same")
                ax.plot(
                    epochs,
                    smooth,
                    label="Smoothed Sinkhorn",
                    color="purple",
                    linewidth=2,
                )
            ax.set_title("Sinkhorn Loss (smoothed)")
            ax.set_xlabel("Epoch")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
            ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_latent_interpolation(
    generator: nn.Module,
    noise_dim: int = 2,
    steps: int = 25,
    title_suffix: str = "",
    real_samples: Optional[np.ndarray] = None,
    latent_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    data_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    cloud_size: int = 512,
) -> None:
    generator.eval()
    with torch.no_grad():
        z_start = torch.randn(noise_dim, device=device)
        z_end = torch.randn(noise_dim, device=device)
        weights = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
        z_path = (1.0 - weights) * z_start + weights * z_end
        path_samples = generator(z_path).cpu().numpy()
        z_cloud = torch.randn(cloud_size, noise_dim, device=device)
        data_cloud = generator(z_cloud).cpu().numpy()
    generator.train()
    z_path_np = z_path.cpu().numpy()
    z_cloud_np = z_cloud.cpu().numpy()
    if latent_limits is None:
        latent_limits = _compute_axis_limits(
            np.vstack([z_cloud_np[:, :2], z_path_np[:, :2]])
        )
    if data_limits is None:
        composite = data_cloud[:, :2]
        if real_samples is not None:
            composite = np.vstack([composite, real_samples[:, :2]])
        composite = np.vstack([composite, path_samples[:, :2]])
        data_limits = _compute_axis_limits(composite)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(
        z_cloud_np[:, 0],
        z_cloud_np[:, 1],
        s=10,
        alpha=0.25,
        color="lightgray",
        label="Latent Points",
    )
    axes[0].plot(
        z_path_np[:, 0],
        z_path_np[:, 1],
        "-o",
        markersize=4,
        color="tab:purple",
        label="Interpolation path",
    )
    axes[0].scatter(
        [z_path_np[0, 0], z_path_np[-1, 0]],
        [z_path_np[0, 1], z_path_np[-1, 1]],
        c=["green", "red"],
        s=70,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[0].set_xlabel("z₁")
    axes[0].set_ylabel("z₂")
    axes[0].set_title("Latent Space Path" + title_suffix)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis("equal")
    axes[0].set_ylim(latent_limits[1])
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].scatter(
        data_cloud[:, 0],
        data_cloud[:, 1],
        s=10,
        alpha=0.25,
        color="silver",
        label="Generated Points",
    )
    if real_samples is not None:
        axes[1].scatter(
            real_samples[:, 0],
            real_samples[:, 1],
            s=10,
            alpha=0.15,
            color="lightgray",
            label="Real data",
        )
    axes[1].plot(
        path_samples[:, 0],
        path_samples[:, 1],
        "-o",
        markersize=4,
        color="tab:orange",
        label="Path image",
    )
    axes[1].scatter(
        [path_samples[0, 0], path_samples[-1, 0]],
        [path_samples[0, 1], path_samples[-1, 1]],
        c=["green", "red"],
        s=70,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")
    axes[1].set_title("Data Space Trajectory")
    axes[1].grid(True, alpha=0.3)
    axes[1].axis("equal")
    axes[1].set_ylim(data_limits[1])
    axes[1].legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()


def load_beams2d_projection(
    n_samples: int = 4000, seed: int = RNG_SEED, n_components: int = 2
):
    try:
        from engibench.problems.beams2d.v0 import Beams2D
    except Exception as exc:
        raise ImportError(
            "Could not import EngiBench. Install it with `pip install engibench` and restart the kernel.\n"
        ) from exc
    problem = Beams2D()
    data = problem.dataset
    designs = np.array(data["train"]["optimal_design"][:n_samples])
    designs = designs.reshape(designs.shape[0], -1)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    embedding = pca.fit_transform(designs)
    return embedding.astype(np.float32), pca
