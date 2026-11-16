import csv
import math
import re
from pathlib import Path
from configurations.configuration import METRIC_DIR

METRIC_DIR = Path(METRIC_DIR)

OUTPUT_PATH = (METRIC_DIR / "model_comparison.csv")

MODELS = [
    "Vanilla AE",
    "Sparse AE",
    "VAE",
    "Vanilla GAN",
    "DCGAN",
    "DDPM"
]

MODEL_ALIASES = {
    "vanilla_ae": "Vanilla AE",
    "ae": "Vanilla AE",
    "sparse_ae": "Sparse AE",
    "vae": "VAE",
    "vanilla_gan": "Vanilla GAN",
    "gan": "Vanilla GAN",
    "dcgan": "DCGAN",
    "ddpm": "DDPM"
}

METRIC_ALIASES = {
    "test_mse": [
        "test_mse", "mse", "reconstruction_mse"
    ],
    "test_psnr": [
        "test_psnr", "psnr", "psnr_db"
    ],
    "test_ssim": [
        "test_ssim", "ssim"
    ],
    "fid_score": [
        "fid_score", "fid"
    ],
    "avg_epoch_time_seconds": [
        "avg_epoch_time_seconds",
        "training_time_per_epoch_seconds",
        "avg_training_time_per_epoch"
    ],
    "inference_latency_ms_batch": [
        "inference_latency_ms_batch",
        "inference_latency_ms",
        "latency_ms_batch"
    ],
    "epochs_completed": [
        "epochs_completed", "epochs", "epoch_count"
    ]
}

def normalize(value):

    return re.sub(
        r"[\s\-]+",
        "_",
        str(value).strip().lower()
    )

def get_model_name(value):

    if not value:
        return None

    key = normalize(value)
    return MODEL_ALIASES.get(key)

def get_value(row, aliases):

    normalized = {
        normalize(key): value
        for key, value in row.items()
    }

    for alias in aliases:
        value = normalized.get(normalize(alias))
        if value not in (None, ""):
            return value

    return None

def to_number(value):
    if value is None:
        return None

    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (ValueError, TypeError):
        pass

    return None

results = {}

for model in MODELS:

    results[model] = {
        "model": model,
        "test_mse": None,
        "test_psnr": None,
        "test_ssim": None,
        "fid_score": None,
        "avg_epoch_time_seconds": None,
        "inference_latency_ms_batch": None,
        "epochs_completed": None
    }

csv_files = [
    file
    for file in METRIC_DIR.glob("*.csv")
    if file.name != OUTPUT_PATH.name
]

for file_path in csv_files:

    if "history" in file_path.stem.lower():
        continue

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    for row in rows:

        model_value = get_value(
            row,
            ["model", "model_name", "architecture"]
        )

        model = get_model_name(model_value)

        if model is None:
            stem = normalize(file_path.stem)
            model = MODEL_ALIASES.get(stem)

        if model is None:
            continue

        for metric, aliases in METRIC_ALIASES.items():

            value = to_number(get_value(row, aliases))

            if value is not None:
                results[model][metric] = value

for file_path in METRIC_DIR.glob("*_history.csv"):

    model_key = normalize(file_path.stem.replace("_history", ""))

    model = MODEL_ALIASES.get(model_key)

    if model is None:
        continue

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:
        continue

    target = results[model]
    target["epochs_completed"] = len(rows)

    epoch_times = [
        to_number(row.get("epoch_time_seconds"))
        for row in rows
    ]

    epoch_times = [
        value for value in epoch_times
        if value is not None
    ]

    if epoch_times:
        target["avg_epoch_time_seconds"] = (
            sum(epoch_times) / len(epoch_times)
        )

#N/A Metrics
#AE, Sparse AE, VAE
for model in [
    "Vanilla GAN",
    "DCGAN",
    "Vanilla GAN",
    "DDPM"
]:
    results[model]["test_mse"] = None
    results[model]["test_psnr"] = None
    results[model]["test_ssim"] = None

for model in ["Vanilla AE", "Sparse AE"]:
    results[model]["fid_score"] = None

#Save

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fields = [
    "model",
    "test_mse",
    "test_psnr",
    "test_ssim",
    "fid_score",
    "avg_epoch_time_seconds",
    "inference_latency_ms_batch",
    "epochs_completed"
]

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()

    for model in MODELS:
        row = results[model]
        writer.writerow({
            key: (
                "N/A"
                if row[key] is None
                else row[key]
            )
            for key in fields
        })

print("=" * 60)
print("MASTER MODEL COMPARISON COMPLETED")
print("=" * 60)

print(f"Models combined: {len(MODELS)}")
print(f"Output: {OUTPUT_PATH}")