"""Generate a presentation-quality loss curve for the trained captioning model.

Run:
    python plot_loss.py

Output:
    assets/loss_curve.png
"""
import os
import matplotlib.pyplot as plt

# Per-epoch validation loss (from README — best epoch is 12)
epochs    = list(range(1, 21))
val_loss  = [
    5.1747, 4.1907, 3.7755, 3.4815, 3.3075,
    3.2293, 3.1301, 3.0613, 3.0237, 3.0123,
    2.9936, 2.9827, 2.9895, 3.0294, 3.0186,
    3.0274, 3.1100, 3.1065, 3.1386, 3.1152,
]

best_epoch = epochs[val_loss.index(min(val_loss))]
best_loss  = min(val_loss)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#222",
    "xtick.color":      "#222",
    "ytick.color":      "#222",
})

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)

# Main curve
ax.plot(epochs, val_loss, color="#6366f1", linewidth=2.4,
        marker="o", markersize=6, markerfacecolor="white",
        markeredgewidth=2, label="Validation Loss")

# Highlight best epoch
ax.scatter([best_epoch], [best_loss], s=180, color="#10b981",
           zorder=5, edgecolor="white", linewidth=2.5,
           label=f"Best (epoch {best_epoch}, loss {best_loss:.4f})")

# Shaded overfit region
ax.axvspan(best_epoch + 0.5, 20.5, alpha=0.08, color="#ef4444")
ax.text(16.5, 5.0, "Overfitting", color="#ef4444",
        fontsize=10, fontweight="bold", ha="center")

# Annotation arrow on best
ax.annotate(
    f"Best checkpoint\nval_loss = {best_loss:.4f}\nPPL = {2.7183**best_loss:.2f}",
    xy=(best_epoch, best_loss),
    xytext=(best_epoch + 2, best_loss + 0.8),
    fontsize=10, color="#10b981",
    arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.5),
)

ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
ax.set_ylabel("Cross-Entropy Loss", fontsize=13, fontweight="bold")
ax.set_title("Training Progress — Flickr8k Captioning Model",
             fontsize=15, fontweight="bold", pad=18, color="#111")

ax.set_xticks(range(1, 21, 2))
ax.set_xlim(0.5, 20.5)
ax.set_ylim(2.7, 5.5)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="upper right", frameon=False, fontsize=11)

# Footer
fig.text(0.5, 0.01,
         "20 epochs · AdamW lr=3e-4 · batch=64 · ~53s/epoch on H100",
         ha="center", fontsize=9, color="#666", style="italic")

os.makedirs("assets", exist_ok=True)
out = "assets/loss_curve.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
