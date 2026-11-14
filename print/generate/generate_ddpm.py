import sys
from pathlib import Path
from types import MethodType
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations.configuration import DEVICE, DDPM_TIMESTEPS
from models.ddpm import DDPM

device = torch.device(DEVICE)

CHECKPOINT_PATH = (PROJECT_ROOT/ "results"/ "checkpoints"/ "ddpm_best.pth")

OUTPUT_DIR = (PROJECT_ROOT/ "results"/ "visualizations"/ "generated_img")

NUM_SAMPLES = 16

def corrected_p_sample(self, x, t):

    beta_t = self.betas[t].view(-1, 1, 1, 1)
    alpha_t = self.alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

    predicted_noise = self.predict_noise(x, t)

    mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t))* predicted_noise) / torch.sqrt(alpha_t)

    if t[0].item() == 0:
        return mean

    noise = torch.randn_like(x)
    return mean + torch.sqrt(beta_t) * noise

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = DDPM(timesteps=DDPM_TIMESTEPS).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH,map_location=device,weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.p_sample = MethodType(corrected_p_sample,model)
    model.eval()

    print("DDPM checkpoint loaded successfully")
    print("Best Epoch:", checkpoint["epoch"])
    print("Validation Loss:", checkpoint["val_loss"])

    return model


def generate_images(model):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    print(f"\nGenerating {NUM_SAMPLES} images")

    with torch.inference_mode():
        generated_images = model.sample(num_samples=NUM_SAMPLES,device=device)
    
    output_path = OUTPUT_DIR / "ddpm_generated_16.png"

    save_image(
        generated_images.cpu(),
        str(output_path),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print("Image generation completed")
    print("Generated images:", output_path)

    return generated_images

def visualize_images(generated_images):

    images = generated_images.cpu().clamp(-1, 1)
    images = (images + 1) / 2

    fig, axes = plt.subplots(4, 4,figsize=(8, 8))

    for i, ax in enumerate(axes.flat):

        ax.imshow(images[i, 0],cmap="gray",vmin=0,vmax=1)
        ax.set_title(f"Generated {i + 1}")
        ax.axis("off")

    plt.suptitle("DDPM Generated Fashion-MNIST Images")
    plt.tight_layout()
    plt.show()

def main():
    print("Device:", device)
    print("DDPM Timesteps:", DDPM_TIMESTEPS)

    model = load_model()
    generated_images = generate_images(model)
    visualize_images(generated_images)
    print("\nDDPM generation completed successfully")

if __name__ == "__main__":
    main()
