import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score


def plot_training_loss(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#378ADD", linewidth=2, label="Train Loss")
    ax.plot(epochs, test_losses, color="#E24B4A", linewidth=2, label="Test Loss", linestyle="--")
    ax.fill_between(epochs, train_losses, test_losses, alpha=0.1, color="#E24B4A")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title("Training vs Test Loss Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/01_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/01_training_loss.png")


def plot_confusion_matrix(y_test, test_probs):
    test_preds = (test_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["Low Risk", "High Risk"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]}\n({cm_norm[i][j]:.1%})",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm_norm[i][j] > 0.5 else "black")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig("outputs/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/02_confusion_matrix.png")


def plot_roc_curve(y_test, test_probs):
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    auc_score = roc_auc_score(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1D9E75", linewidth=2.5,
            label=f"Neural Network (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1,
            linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1D9E75")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/03_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/03_roc_curve.png")


def plot_risk_distribution(y_test, test_probs):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(test_probs[y_test == 0], bins=bins, color="#1D9E75",
            alpha=0.6, label="Actual Low Risk", density=True)
    ax.hist(test_probs[y_test == 1], bins=bins, color="#E24B4A",
            alpha=0.6, label="Actual High Risk", density=True)
    ax.axvline(0.5, color="black", linewidth=1.5,
               linestyle="--", label="Decision Threshold (0.5)")
    ax.set_xlabel("Predicted Disease Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Predicted Risk Distribution by Actual Class",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/04_risk_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/04_risk_distribution.png")


if __name__ == "__main__":
    print("Run main.py instead of visualize.py directly.")