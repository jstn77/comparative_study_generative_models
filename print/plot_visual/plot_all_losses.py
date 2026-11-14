import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations.configuration import METRIC_DIR, RESULT_DIR

METRIC_PATH = Path(METRIC_DIR)
PLOT_DIR = Path(RESULT_DIR) / "visualizations" / "plots"

MODEL_CONFIG = {
    "vanilla_ae": {
        "title": "Vanilla Autoencoder",
        "file": "vanilla_ae_history.csv",
        "series": [
            ("train_loss", "Train Loss"),
            ("val_loss", "Validation Loss"),
        ],
    },
    "sparse_ae": {
        "title": "Sparse Autoencoder",
        "file": "sparse_ae_history.csv",
        "series": [
            ("train_loss", "Train Loss"),
            ("val_loss", "Validation Loss"),
        ],
    },
    "vae": {
        "title": "Variational Autoencoder",
        "file": "vae_history.csv",
        "series": [
            ("train_loss", "Train Total Loss"),
            ("val_loss", "Validation Total Loss"),
        ],
    },
    "vanilla_gan": {
        "title": "Vanilla GAN",
        "file": "vanilla_gan_history.csv",
        "series": [
            ("train_g_loss", "Train Generator Loss"),
            ("train_d_loss", "Train Discriminator Loss"),
            ("val_g_loss", "Validation Generator Loss"),
            ("val_d_loss", "Validation Discriminator Loss"),
        ],
    },
    "dcgan": {
        "title": "DCGAN",
        "file": "dcgan_history.csv",
        "series": [
            ("train_g_loss", "Train Generator Loss"),
            ("train_d_loss", "Train Discriminator Loss"),
            ("val_g_loss", "Validation Generator Loss"),
            ("val_d_loss", "Validation Discriminator Loss"),
        ],
    },
}

def read_history(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"History CSV tidak ditemukan: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV kosong: {csv_path}")
    return rows

def plot_one_model(model_key, config):
    csv_path = METRIC_PATH / config["file"]
    rows = read_history(csv_path)
    epochs = [int(row["epoch"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    for column, label in config["series"]:
        if column not in rows[0]:
            raise KeyError(
                f"Kolom '{column}' tidak ada di {csv_path.name}. "
                f"Kolom tersedia: {list(rows[0].keys())}"
            )
        values = [float(row[column]) for row in rows]
        ax.plot(epochs, values, linewidth=2, label=label)

    ax.set_title(f"{config['title']} - Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = PLOT_DIR / f"{model_key}_loss_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")

def plot_combined_overview():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    overview_models = [
        ("vanilla_ae", "Vanilla AE"),
        ("sparse_ae", "Sparse AE"),
        ("vae", "VAE"),
    ]

    for ax, (key, title) in zip(axes, overview_models):
        config = MODEL_CONFIG[key]
        rows = read_history(METRIC_PATH / config["file"])
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(epochs, [float(r["train_loss"]) for r in rows], label="Train")
        ax.plot(epochs, [float(r["val_loss"]) for r in rows], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Training Loss Comparison: AE, Sparse AE, and VAE")
    fig.tight_layout()
    output_path = PLOT_DIR / "ae_sparse_ae_vae_loss_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("TRAINING LOSS VISUALIZATION - WITHOUT DDPM")
    print("=" * 68)

    for model_key, config in MODEL_CONFIG.items():
        plot_one_model(model_key, config)

    plot_combined_overview()
    print("\nSelesai. Grafik tersimpan di:", PLOT_DIR)

if __name__ == "__main__":
    main()
