import json
import os
import matplotlib.pyplot as plt

LOG_DIR = "logs"

FILES = {
    "EGP": "egp_depth4_seed0.json",
    "EP-EGP": "ep_egp_depth4_seed0.json",
    "F-EGP": "f_egp_depth4_seed0.json",
}

plt.figure(figsize=(7, 5))

for label, fname in FILES.items():
    path = os.path.join(LOG_DIR, fname)
    with open(path, "r") as f:
        data = json.load(f)

    val_curve = data["curves"]["val_acc"]
    epochs = range(1, len(val_curve) + 1)

    plt.plot(epochs, val_curve, label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy (depth=4, layers=7, seed=0)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("treeneighbor_generalization_depth4layer7_seed0.png", dpi=300)
plt.show()
