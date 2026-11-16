import os
import sys
import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations.configuration import (
    DEVICE,
    IMAGE_DIM,
    IMAGE_SIZE,
    HIDDEN_DIM,
    LATENT_DIM,
    GAN_NOISE_DIM,
    DCGAN_NOISE_DIM,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    RESULT_DIR,
)

from data.dataset import get_data_loaders
from models.vanilla_ae import Autoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.gan import VanillaGenerator
from models.dcgan import DCGANGenerator

SEED = 42

DRESS_CLASS = 3
SNEAKER_CLASS = 7

CLASS_NAMES = {
    DRESS_CLASS: "Dress",
    SNEAKER_CLASS: "Sneaker",
}

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

GAN_SEARCH_SAMPLES = 10000

SEARCH_BATCH_SIZE = 512

OUTPUT_DIR = (
    Path(RESULT_DIR)
    / "visualizations"
    / "interpolations"
)

CHECKPOINT_PATH = Path(CHECKPOINT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(path, model, state_key="model_state_dict"):

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if state_key not in checkpoint:
        raise KeyError(
            f"Key '{state_key}' tidak ditemukan dalam "
            f"{path.name}. Keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(checkpoint[state_key])

    model.to(DEVICE)
    model.eval()

    print(f"[LOADED] {path.name}")

    return model

def select_endpoint_images(test_loader):

    print("\n[STEP 1] Selecting original endpoint images")

    dress_image = None
    sneaker_image = None

    with torch.no_grad():

        for images, labels in test_loader:

            for i in range(len(labels)):

                label = int(labels[i].item())

                if label == DRESS_CLASS and dress_image is None:
                    dress_image = images[i:i+1].clone()

                elif label == SNEAKER_CLASS and sneaker_image is None:
                    sneaker_image = images[i:i+1].clone()

                if dress_image is not None and sneaker_image is not None:
                    break

            if dress_image is not None and sneaker_image is not None:
                break

    if dress_image is None or sneaker_image is None:
        raise RuntimeError(
            "Tidak berhasil menemukan sampel Dress dan Sneaker."
        )

    dress_image = dress_image.to(DEVICE)
    sneaker_image = sneaker_image.to(DEVICE)

    print("Dress label:", DRESS_CLASS)
    print("Sneaker label:", SNEAKER_CLASS)

    return dress_image, sneaker_image

def load_models():

    print("\n[STEP 2] Loading five generative models")

    #Vanilla AE
    vanilla_ae = Autoencoder().to(DEVICE)

    vanilla_ae = load_checkpoint(
        CHECKPOINT_PATH / "vanilla_ae_best.pth",
        vanilla_ae
    )

    #Sparse AE
    sparse_ae = SparseAutoencoder().to(DEVICE)

    sparse_ae = load_checkpoint(
        CHECKPOINT_PATH / "sparse_ae_best.pth",
        sparse_ae
    )

    #VAE
    vae = VAE().to(DEVICE)

    vae = load_checkpoint(
        CHECKPOINT_PATH / "vae_best.pth",
        vae
    )

    vanilla_gan = VanillaGenerator().to(DEVICE)

    vanilla_gan = load_checkpoint(
        CHECKPOINT_PATH / "vanilla_gan_best.pth",
        vanilla_gan,
        state_key="generator_state_dict"
    )

    dcgan = DCGANGenerator().to(DEVICE)

    dcgan = load_checkpoint(
        CHECKPOINT_PATH / "dcgan_best.pth",
        dcgan,
        state_key="generator_state_dict"
    )

    return {
        "Vanilla AE": vanilla_ae,
        "Sparse AE": sparse_ae,
        "VAE": vae,
        "Vanilla GAN": vanilla_gan,
        "DCGAN": dcgan,
    }

def interpolate_latent(z_a, z_b):

    results = []

    for alpha in ALPHAS:

        z_interpolated = (
            (1.0 - alpha) * z_a
            + alpha * z_b
        )

        results.append(z_interpolated)

    return results


@torch.no_grad()
def generate_ae_interpolation(model, image_a, image_b):
    model.eval()

    z_a = model.encode(image_a)
    z_b = model.encode(image_b)

    latent_points = interpolate_latent(z_a, z_b)

    generated_images = []

    for z in latent_points:

        reconstructed = model.decode(z)

        generated_images.append(
            reconstructed.detach().cpu()
        )

    return torch.cat(generated_images, dim=0)


@torch.no_grad()
def generate_vae_interpolation(model, image_a, image_b):

    model.eval()
    mu_a, logvar_a = model.encode(image_a)
    mu_b, logvar_b = model.encode(image_b)

    z_a = mu_a
    z_b = mu_b

    latent_points = interpolate_latent(z_a, z_b)

    generated_images = []

    for z in latent_points:

        reconstructed = model.decode(z)

        generated_images.append(
            reconstructed.detach().cpu()
        )

    return torch.cat(generated_images, dim=0)

class MLPClassifier(nn.Module):
    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(IMAGE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, NUM_CLASSES)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(x.size(0), -1)

        h1 = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h1))

        return self.fc3(z)


def load_mlp_classifier():

    print("\n[STEP 3] Loading MLP classifier")

    classifier = MLPClassifier().to(DEVICE)

    classifier = load_checkpoint(
        CHECKPOINT_PATH / "mlp_best.pth",
        classifier
    )

    return classifier

@torch.no_grad()
def search_gan_latent(
    generator,
    classifier,
    target_class,
    noise_dim,
    is_dcgan=False
):

    print(
        f"\nSearching latent vector for "
        f"{CLASS_NAMES[target_class]}..."
    )

    generator.eval()
    classifier.eval()

    best_probability = -1.0
    best_latent = None
    best_image = None

    total_searched = 0

    while total_searched < GAN_SEARCH_SAMPLES:

        current_batch = min(
            SEARCH_BATCH_SIZE,
            GAN_SEARCH_SAMPLES - total_searched
        )

        if is_dcgan:

            z = torch.randn(
                current_batch,
                noise_dim,
                1,
                1,
                device=DEVICE
            )

        else:

            z = torch.randn(
                current_batch,
                noise_dim,
                device=DEVICE
            )

        generated = generator(z)

        logits = classifier(generated)

        probabilities = torch.softmax(logits, dim=1)

        target_probabilities = probabilities[:, target_class]

        batch_best_index = torch.argmax(
            target_probabilities
        )

        batch_best_probability = (
            target_probabilities[batch_best_index].item()
        )

        if batch_best_probability > best_probability:

            best_probability = batch_best_probability

            best_latent = (
                z[batch_best_index:batch_best_index+1]
                .detach()
                .clone()
            )

            best_image = (
                generated[batch_best_index:batch_best_index+1]
                .detach()
                .clone()
            )

        total_searched += current_batch

        if total_searched % 2048 == 0:
            print(
                f"  Searched: {total_searched}/"
                f"{GAN_SEARCH_SAMPLES}"
            )

    print(
        f"Best predicted probability for "
        f"{CLASS_NAMES[target_class]}: "
        f"{best_probability:.4f}"
    )

    return best_latent, best_image, best_probability

@torch.no_grad()
def generate_gan_interpolation(
    generator,
    classifier,
    noise_dim,
    is_dcgan=False
):

    generator.eval()

    z_a, image_a, confidence_a = search_gan_latent(
        generator,
        classifier,
        DRESS_CLASS,
        noise_dim,
        is_dcgan
    )

    z_b, image_b, confidence_b = search_gan_latent(
        generator,
        classifier,
        SNEAKER_CLASS,
        noise_dim,
        is_dcgan
    )

    latent_points = interpolate_latent(z_a, z_b)

    generated_images = []

    for z in latent_points:

        generated = generator(z)

        generated_images.append(
            generated.detach().cpu()
        )

    return (
        torch.cat(generated_images, dim=0),
        confidence_a,
        confidence_b,
        image_a.detach().cpu(),
        image_b.detach().cpu()
    )

def to_display_range(images):

    # Models use Tanh output: [-1, 1]
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def save_model_interpolation(
    model_name,
    images,
    output_dir
):

    images = to_display_range(images)

    fig, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(15, 3)
    )

    for i, alpha in enumerate(ALPHAS):

        axes[i].imshow(
            images[i, 0].numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i].set_title(
            f"α = {alpha:.1f}",
            fontsize=10
        )

        axes[i].axis("off")

    fig.suptitle(
        f"{model_name} - Latent Space Interpolation",
        fontsize=13
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_interpolation.png"
    )

    path = output_dir / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("[SAVED]", path)


def save_combined_figure(all_results, output_dir):

    print("\n[STEP 5] Creating combined comparison figure")

    model_names = list(all_results.keys())

    fig, axes = plt.subplots(
        len(model_names),
        len(ALPHAS),
        figsize=(15, 11)
    )

    for row, model_name in enumerate(model_names):

        images = to_display_range(
            all_results[model_name]
        )

        for col, alpha in enumerate(ALPHAS):

            axes[row, col].imshow(
                images[col, 0].numpy(),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(
                    f"α = {alpha:.1f}",
                    fontsize=10
                )

            if col == 0:
                axes[row, col].set_ylabel(
                    model_name,
                    fontsize=10
                )

    fig.suptitle(
        "Linear Latent Space Interpolation - Fashion-MNIST",
        fontsize=15
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "figure4_latent_space_interpolation.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("[SAVED]", output_path)

def save_search_metrics(search_results, output_dir):

    output_path = output_dir / "gan_latent_search.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model",
            "dress_classifier_probability",
            "sneaker_classifier_probability"
        ])

        for model_name, values in search_results.items():

            writer.writerow([
                model_name,
                values[0],
                values[1]
            ])

    print("[SAVED]", output_path)

def main():

    set_seed()

    print("=" * 70)
    print("FASHION-MNIST LATENT SPACE INTERPOLATION")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Latent dimension:", LATENT_DIM)
    print("GAN noise dimension:", GAN_NOISE_DIM)
    print("DCGAN noise dimension:", DCGAN_NOISE_DIM)
    print("Alpha values:", ALPHAS)

    print("\nLoading Fashion-MNIST test dataset")

    _, _, test_loader = get_data_loaders()

    image_a, image_b = select_endpoint_images(test_loader)

    models = load_models()

    classifier = load_mlp_classifier()

    all_results = {}

    print("\n[MODEL 1] Vanilla AE")

    all_results["Vanilla AE"] = generate_ae_interpolation(
        models["Vanilla AE"],
        image_a,
        image_b
    )

    print("\n[MODEL 2] Sparse AE")

    all_results["Sparse AE"] = generate_ae_interpolation(
        models["Sparse AE"],
        image_a,
        image_b
    )

    print("\n[MODEL 3] VAE")

    all_results["VAE"] = generate_vae_interpolation(
        models["VAE"],
        image_a,
        image_b
    )

    print("\n[MODEL 4] Vanilla GAN")

    (
        gan_images,
        gan_dress_confidence,
        gan_sneaker_confidence,
        gan_dress_image,
        gan_sneaker_image
    ) = generate_gan_interpolation(
        models["Vanilla GAN"],
        classifier,
        GAN_NOISE_DIM,
        is_dcgan=False
    )

    all_results["Vanilla GAN"] = gan_images

    print("\n[MODEL 5] DCGAN")

    (
        dcgan_images,
        dcgan_dress_confidence,
        dcgan_sneaker_confidence,
        dcgan_dress_image,
        dcgan_sneaker_image
    ) = generate_gan_interpolation(
        models["DCGAN"],
        classifier,
        DCGAN_NOISE_DIM,
        is_dcgan=True
    )

    all_results["DCGAN"] = dcgan_images

    print("\n[STEP 4] Saving individual interpolation figures")

    for model_name, images in all_results.items():

        save_model_interpolation(
            model_name,
            images,
            OUTPUT_DIR
        )

    save_combined_figure(
        all_results,
        OUTPUT_DIR
    )

    search_results = {
        "Vanilla GAN": (
            gan_dress_confidence,
            gan_sneaker_confidence
        ),
        "DCGAN": (
            dcgan_dress_confidence,
            dcgan_sneaker_confidence
        )
    }

    save_search_metrics(
        search_results,
        OUTPUT_DIR
    )

    endpoint_images = torch.cat([
        gan_dress_image,
        gan_sneaker_image,
        dcgan_dress_image,
        dcgan_sneaker_image
    ], dim=0)

    endpoint_images = to_display_range(endpoint_images)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5, 5)
    )

    titles = [
        "Vanilla GAN - Dress",
        "Vanilla GAN - Sneaker",
        "DCGAN - Dress",
        "DCGAN - Sneaker"
    ]

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            endpoint_images[i, 0].numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gan_selected_endpoints.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    metadata_path = OUTPUT_DIR / "interpolation_metadata.txt"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("FASHION-MNIST LATENT INTERPOLATION\n")
        file.write("=" * 50 + "\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Classes: Dress ({DRESS_CLASS}), ")
        file.write(f"Sneaker ({SNEAKER_CLASS})\n")

        file.write(f"Alpha values: {ALPHAS}\n")
        file.write(f"GAN search samples: {GAN_SEARCH_SAMPLES}\n")

        file.write("\nGAN CLASSIFIER CONFIDENCE\n")

        for model_name, values in search_results.items():

            file.write(f"\n{model_name}\n")
            file.write(f"Dress: {values[0]:.6f}\n")
            file.write(f"Sneaker: {values[1]:.6f}\n")

    print("\n" + "=" * 70)
    print("LATENT INTERPOLATION COMPLETED")
    print("=" * 70)

    print("\nAll outputs saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- figure4_latent_space_interpolation.png")
    print("- vanilla_ae_interpolation.png")
    print("- sparse_ae_interpolation.png")
    print("- vae_interpolation.png")
    print("- vanilla_gan_interpolation.png")
    print("- dcgan_interpolation.png")
    print("- gan_selected_endpoints.png")
    print("- gan_latent_search.csv")
    print("- interpolation_metadata.txt")


if __name__ == "__main__":
    main()
