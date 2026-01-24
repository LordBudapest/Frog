import os
import json
import matplotlib.pyplot as plt

def load_curve(path):
    with open(path) as f:
        return json.load(f)

# ensure output directory exists
os.makedirs("plots", exist_ok=True)

egp = load_curve("results/peptides_func_seed0_egp_curves.json")
ep  = load_curve("results/peptides_func_seed0_ep-egp_curves.json")
f   = load_curve("results/peptides_func_seed0_f-egp_curves.json")

plt.figure(figsize=(6, 4))

plt.plot(ep["val_ap"], label="EP-EGP", linewidth=2)
plt.plot(f["val_ap"], label="F-EGP", linewidth=2)
plt.plot(egp["val_ap"], label="EGP", linestyle="--", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Validation AP")
plt.ylim(0.45, 0.65)
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("plots/peptides_func_generalization_seed0.png", dpi=300)
plt.show()
