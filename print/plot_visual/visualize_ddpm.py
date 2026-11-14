import os
import pandas as pd
import matplotlib.pyplot as plt

#metrics path
CSV_PATH = "results/metrics/ddpm_history.csv"

#read training result
df = pd.read_csv(CSV_PATH)

print("DDPM TRAINING HISTORY")
print(df.head())
print("\nJumlah epoch:", len(df))
print("Best Train Loss:", df["train_loss"].min())
print("Best Validation Loss:", df["val_loss"].min())

#loss
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"],df["train_loss"],label="Training Loss")
plt.plot(df["epoch"],df["val_loss"],label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("DDPM Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

#save
os.makedirs("results/visualizations", exist_ok=True)

plt.savefig("results/visualizations/ddpm_loss_curve.png",dpi=300)
plt.show()

print("\nSaved")
