import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "fashion_mnist")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

SEED = 42

DATASET_NAME = "FashionMNIST"
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 64
HIDDEN_DIM = 256
GAN_NOISE_DIM = 64
DCGAN_NOISE_DIM = 100
DDPM_TIMESTEPS = 1000

EPOCHS_AE = 30
EPOCHS_SPARSE_AE = 30
EPOCHS_VAE = 30
EPOCHS_GAN = 50
EPOCHS_DCGAN = 20
EPOCHS_DDPM = 20
EPOCHS_MLP = 30

LEARNING_RATE = 0.001
GAN_LEARNING_RATE = 0.0002
DDPM_LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

SPARSITY_LAMBDA = 0.001

NUM_GENERATED_SAMPLES = 16
LATENCY_BATCH_SIZE = 64

print("Loaded successfully!")
print("Device:", DEVICE)