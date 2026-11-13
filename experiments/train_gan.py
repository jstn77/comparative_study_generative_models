import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from configurations.configuration import (
    DEVICE,
    SEED,
    GAN_LEARNING_RATE,
    EPOCHS_GAN,
    GAN_NOISE_DIM,
    BETAS,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.gan import (
    VanillaGenerator,
    VanillaDiscriminator
)

def train_vanilla_gan():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders()

    generator = VanillaGenerator().to(DEVICE)
    discriminator = VanillaDiscriminator().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer_g = Adam(generator.parameters(),lr=GAN_LEARNING_RATE,betas=BETAS)
    optimizer_d = Adam(discriminator.parameters(),lr=GAN_LEARNING_RATE,betas=BETAS)

    best_val_g_loss = float("inf")
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_GAN)

    for epoch in range(EPOCHS_GAN):
        start_time = time.perf_counter()
        
        #training
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_samples = 0

        for images, _ in train_loader:
            images = images.to(DEVICE)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            #train discriminator
            noise = torch.randn(batch_size, GAN_NOISE_DIM, device=DEVICE)
            fake_images = generator(noise)
            real_output = discriminator(images)
            fake_output = discriminator(fake_images.detach())

            real_loss = criterion(real_output,real_labels)
            fake_loss = criterion(fake_output,fake_labels)

            d_loss = real_loss + fake_loss

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            #train generator
            noise = torch.randn(batch_size,GAN_NOISE_DIM,device=DEVICE)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images)

            g_loss = criterion(fake_output,real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            total_g_loss += g_loss.item() * batch_size
            total_d_loss += d_loss.item() * batch_size
            total_samples += batch_size

        train_g_loss = total_g_loss / total_samples
        train_d_loss = total_d_loss / total_samples

        #validation
        generator.eval()
        discriminator.eval()

        val_g_loss = 0.0
        val_d_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                batch_size = images.size(0)
                real_labels = torch.ones(batch_size, 1, device=DEVICE)
                fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

                noise = torch.randn(batch_size,GAN_NOISE_DIM,device=DEVICE)

                fake_images = generator(noise)

                real_output = discriminator(images)
                fake_output = discriminator(fake_images)

                d_loss = (criterion(real_output, real_labels) + criterion(fake_output, fake_labels))
                g_loss = criterion(discriminator(fake_images), real_labels)

                val_g_loss += g_loss.item() * batch_size
                val_d_loss += d_loss.item() * batch_size
                val_samples += batch_size

        val_g_loss /= val_samples
        val_d_loss /= val_samples

        epoch_time = time.perf_counter() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_g_loss": train_g_loss,
            "train_d_loss": train_d_loss,
            "val_g_loss": val_g_loss,
            "val_d_loss": val_d_loss,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS_GAN}] "
            f"| G Loss: {train_g_loss:.4f} "
            f"| D Loss: {train_d_loss:.4f} "
            f"| Val G: {val_g_loss:.4f} "
            f"| Val D: {val_d_loss:.4f} "
            f"| Time: {epoch_time:.2f}s"
        )

        #save best G and best D
        if val_g_loss < best_val_g_loss:
            best_val_g_loss = val_g_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "val_g_loss": val_g_loss,
                    "val_d_loss": val_d_loss
                },
                os.path.join(CHECKPOINT_DIR,"vanilla_gan_best.pth")
            )

    #save history
    history_path = os.path.join(METRIC_DIR,"vanilla_gan_history.csv")

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_g_loss",
                "train_d_loss",
                "val_g_loss",
                "val_d_loss",
                "epoch_time_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining Completed")
    print(f"Best Validation Generator Loss: {best_val_g_loss:.6f}")
    print("Checkpoint saved:", CHECKPOINT_DIR)
    print("Training history saved:", history_path)

if __name__ == "__main__":
    train_vanilla_gan()
