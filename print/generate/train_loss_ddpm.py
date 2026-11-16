import sys
from pathlib import Path
from types import MethodType
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configurations.configuration import (DEVICE,DDPM_TIMESTEPS,)
from models.ddpm import DDPM

SEED = 42

RESULT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

OUTPUT_DIR = (
    RESULT_DIR
    / "visualizations"
    / "figure5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DDPM_CHECKPOINT = CHECKPOINT_DIR / "ddpm_best.pth"

SNAPSHOT_STEPS = [1000, 800, 600, 400, 200, 100, 0]

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def find_history_csv(model_aliases):
    csv_files = list(RESULT_DIR.rglob("*.csv"))
    candidates = []
    for path in csv_files:
        filename = path.stem.lower()
        if any(alias in filename for alias in model_aliases):
            candidates.append(path)

    if not candidates:
        available = "\n".join(
            str(path.relative_to(RESULT_DIR))
            for path in csv_files
        )

        raise FileNotFoundError(
            f"\nHistory CSV tidak ditemukan untuk: {model_aliases}"
            f"\nCSV yang tersedia:\n{available}"
        )

    candidates.sort(
        key=lambda p: (
            "history" not in p.stem.lower()
            and "train" not in p.stem.lower(),
            len(p.parts),
            len(p.name)
        )
    )

    selected = candidates[0]
    print(f"[CSV] {selected}")
    return pd.read_csv(selected)

def normalize_column(name):
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("/", "_")
    )

def find_column(df, aliases, required=True):
    columns = {
        normalize_column(col): col
        for col in df.columns
    }

    for alias in aliases:
        key = normalize_column(alias)
        if key in columns:
            return columns[key]

    for alias in aliases:
        key = normalize_column(alias)
        for normalized, original in columns.items():
            if key in normalized:
                return original

    if required:
        raise ValueError(
            f"Kolom tidak ditemukan. Dicari: {aliases}"
            f"\nKolom tersedia: {list(df.columns)}"
        )
    return None

def get_epoch_values(df):
    epoch_col = find_column(
        df,
        ["epoch", "epochs"],
        required=False
    )
    if epoch_col is not None:
        return df[epoch_col].values

    return np.arange(1, len(df) + 1)

def plot_train_val(ax,df,train_aliases,val_aliases,title,ylabel="Loss"):
    train_col = find_column(
        df,
        train_aliases
    )
    val_col = find_column(
        df,
        val_aliases,
        required=False
    )
    epochs = get_epoch_values(df)
    ax.plot(
        epochs,
        df[train_col].values,
        label="Training Loss",
        linewidth=1.8
    )
    if val_col is not None:
        ax.plot(
            epochs,
            df[val_col].values,
            label="Validation Loss",
            linewidth=1.8,
            linestyle="--"
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

def plot_gan_losses(ax,df,title):
    epoch = get_epoch_values(df)
    g_col = find_column(
        df,
        [
            "g_loss",
            "generator_loss",
            "train_g_loss",
            "train_generator_loss",
            "loss_g",
            "train_loss_g"
        ]
    )
    d_col = find_column(
        df,
        [
            "d_loss",
            "discriminator_loss",
            "train_d_loss",
            "train_discriminator_loss",
            "loss_d",
            "train_loss_d"
        ]
    )
    ax.plot(
        epoch,
        df[g_col].values,
        label="Generator Loss",
        linewidth=1.7
    )
    ax.plot(
        epoch,
        df[d_col].values,
        label="Discriminator Loss",
        linewidth=1.7
    )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Adversarial Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

def generate_loss_figure():
    print("\n[STEP 1] Loading training histories")
    ae_df = find_history_csv(["vanilla_ae"])
    sparse_df = find_history_csv(["sparse_ae"])
    vae_df = find_history_csv(["vae"])
    gan_df = find_history_csv(["vanilla_gan"])
    dcgan_df = find_history_csv(["dcgan"])
    ddpm_df = find_history_csv(["ddpm"])

    fig, axes = plt.subplots(3,2,figsize=(15, 14))

    plot_train_val(
        axes[0, 0],
        ae_df,
        ["train_loss", "training_loss", "loss"],
        ["val_loss", "validation_loss"],
        "Vanilla AE - Reconstruction Loss"
    )

    plot_train_val(
        axes[0, 1],
        sparse_df,
        ["train_loss", "training_loss", "loss"],
        ["val_loss", "validation_loss"],
        "Sparse AE - Reconstruction Loss"
    )

    plot_train_val(
        axes[1, 0],
        vae_df,
        [
            "train_recon_loss",
            "train_reconstruction_loss",
            "recon_loss",
            "reconstruction_loss"
        ],
        [
            "val_recon_loss",
            "val_reconstruction_loss",
            "val_reconstruction"
        ],
        "VAE - Reconstruction Loss"
    )

    plot_gan_losses(
        axes[1, 1],
        gan_df,
        "Vanilla GAN - Generator and Discriminator Loss"
    )

    plot_gan_losses(
        axes[2, 0],
        dcgan_df,
        "DCGAN - Generator and Discriminator Loss"
    )

    plot_train_val(
        axes[2, 1],
        ddpm_df,
        [
            "train_loss",
            "train_noise_loss",
            "train_noise_prediction_loss",
            "loss"
        ],
        [
            "val_loss",
            "val_noise_loss",
            "val_noise_prediction_loss"
        ],
        "DDPM - Noise Prediction Loss"
    )
    fig.suptitle(
        "Training Loss Curves of Six Generative Models",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = (
        OUTPUT_DIR
        / "figure5_training_loss_curves.png"
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    print("[SAVED]", output_path)

@torch.no_grad()
def corrected_p_sample(self, x, t):
    beta_t = self.betas[t].view(-1, 1, 1, 1)
    alpha_t = self.alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
    predicted_noise = self.predict_noise(x, t)
    mean = (
        x
        - (
            beta_t
            / torch.sqrt(1.0 - alpha_bar_t)
        ) * predicted_noise
    ) / torch.sqrt(alpha_t)

    if t[0].item() == 0:
        return mean
    noise = torch.randn_like(x)
    return mean + torch.sqrt(beta_t) * noise

def load_ddpm():
    print("\n[STEP 2] Loading DDPM checkpoint")
    if not DDPM_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"DDPM checkpoint tidak ditemukan: "
            f"{DDPM_CHECKPOINT}"
        )

    model = DDPM(timesteps=DDPM_TIMESTEPS).to(DEVICE)

    checkpoint = torch.load(
        DDPM_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.p_sample = MethodType(corrected_p_sample,model)

    model.eval()

    print("DDPM loaded successfully")
    print("Best epoch:", checkpoint.get("epoch"))
    print("Validation loss:", checkpoint.get("val_loss"))
    return model

@torch.no_grad()
def generate_ddpm_trajectory(model):
    print("\n[STEP 3] Generating DDPM denoising trajectory")

    x = torch.randn(1,1,28,28,device=DEVICE)
    snapshots = {}

    snapshots[DDPM_TIMESTEPS] = x.detach().cpu().clone()

    for timestep in reversed(range(DDPM_TIMESTEPS)):
        t = torch.full(
            (1,),
            timestep,
            device=DEVICE,
            dtype=torch.long
        )
        x = model.p_sample(x, t)
        current_step = timestep
        if current_step in SNAPSHOT_STEPS:
            snapshots[current_step] = (
                x.detach().cpu().clone()
            )

        if current_step % 200 == 0:
            print(f"Reverse diffusion step: {current_step}")

    snapshots[0] = x.detach().cpu().clone()
    return snapshots

def normalize_snapshot(image, is_final=False):
    image = image.squeeze().numpy()
    if is_final:
        image = (image + 1.0) / 2.0
        return np.clip(image, 0, 1)

    minimum = image.min()
    maximum = image.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(image)

    return (image - minimum) / (maximum - minimum)

def save_ddpm_trajectory(snapshots):
    print("\n[STEP 4] Saving DDPM trajectory figure")
    steps = [1000,800,600,400,200,100,0]
    fig, axes = plt.subplots(
        1,
        len(steps),
        figsize=(16, 3)
    )

    for index, step in enumerate(steps):
        image = snapshots[step]
        display_image = normalize_snapshot(
            image,
            is_final=(step == 0)
        )
        axes[index].imshow(
            display_image,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        if step == 1000:
            label = r"$x_T$ (Initial Noise)"

        elif step == 0:
            label = r"$x_0$ (Final Image)"

        else:
            label = rf"$x_{{{step}}}$"

        axes[index].set_title(
            label,
            fontsize=11
        )
        axes[index].axis("off")
    fig.suptitle(
        "DDPM Reverse Diffusion Denoising Trajectory",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    output_path = (
        OUTPUT_DIR
        / "figure5_ddpm_denoising_trajectory.png"
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    print("[SAVED]", output_path)

    torch.save(
        snapshots,
        OUTPUT_DIR / "ddpm_trajectory_snapshots.pth"
    )

def main():
    set_seed()

    print("=" * 65)
    print("TRAINING LOSS AND DDPM DENOISING")
    print("=" * 65)

    print("Device:", DEVICE)

    generate_loss_figure()
    ddpm = load_ddpm()

    snapshots = generate_ddpm_trajectory(ddpm)

    save_ddpm_trajectory(snapshots)

    print("\n" + "=" * 65)
    print("FIGURE 5 COMPLETED")
    print("=" * 65)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- figure5_training_loss_curves.png")
    print("- figure5_ddpm_denoising_trajectory.png")
    print("- ddpm_trajectory_snapshots.pth")

if __name__ == "__main__":
    main()
