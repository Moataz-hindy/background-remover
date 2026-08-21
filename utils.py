import torch
import matplotlib.pyplot as plt

def train_step(model, data_loader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}")
    return train_loss


def val_step(model, data_loader, loss_fn, device):
    val_loss, val_iou, val_dice = 0, 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            val_pred = model(X)
            probs = torch.sigmoid(val_pred)
            pred = probs > 0.4

            val_loss += loss_fn(val_pred, y).item()
            iou, dice = segmentation_metrics(pred, y)
            val_iou += iou.mean().item()
            val_dice += dice.mean().item()

        val_loss /= len(data_loader)
        val_iou /= len(data_loader)
        val_dice /= len(data_loader)

        print(f"Val loss: {val_loss:.5f} | Val iou: {val_iou:.5f} | Val dice: {val_dice:.5f}")
    return val_loss, val_iou, val_dice


def segmentation_metrics(pred, target):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    dice_denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    iou = (intersection + 1e-8) / (union + 1e-8)
    dice = (2 * intersection + 1e-8) / (dice_denominator + 1e-8)
    return iou, dice


def evaluate_thresholds(model, data_loader, device):
    model.eval()
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = {}

    with torch.inference_mode():
        for threshold in thresholds:
            total_iou = 0
            total_dice = 0

            for X, y in data_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X)
                probs = torch.sigmoid(logits)
                pred = (probs > threshold).float()

                intersection = (pred * y).sum(dim=(1, 2, 3))
                union = ((pred + y) > 0).float().sum(dim=(1, 2, 3))
                dice_denominator = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))

                iou = (intersection + 1e-8) / (union + 1e-8)
                dice = (2 * intersection + 1e-8) / (dice_denominator + 1e-8)

                total_iou += iou.mean().item()
                total_dice += dice.mean().item()

            avg_iou = total_iou / len(data_loader)
            avg_dice = total_dice / len(data_loader)

            results[threshold] = {
                "IoU": avg_iou,
                "Dice": avg_dice
            }

            print(
                f"Threshold: {threshold:.1f} | "
                f"IoU: {avg_iou:.4f} | "
                f"Dice: {avg_dice:.4f}"
            )

    return results


def visualize_predictions(
    model,
    dataloader,
    device,
    num_samples=8,
    threshold=0.4
):
    model.eval()

    X, y = next(iter(dataloader))
    X = X.to(device)
    y = y.to(device)

    with torch.inference_mode():
        logits = model(X)
        probs = torch.sigmoid(logits)
        pred_masks = (probs > threshold).float()

    X = X.cpu()
    y = y.cpu()
    probs = probs.cpu()
    pred_masks = pred_masks.cpu()

    num_samples = min(num_samples, X.size(0))

    fig, axes = plt.subplots(
        num_samples,
        4,
        figsize=(16, 3 * num_samples)
    )

    if num_samples == 1:
        axes = axes.reshape(1, 4)

    for i in range(num_samples):
        image = X[i].permute(1, 2, 0)
        true_mask = y[i].squeeze()
        probability = probs[i].squeeze()
        pred_mask = pred_masks[i].squeeze()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(probability, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Probability")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_mask, cmap="gray")
        axes[i, 3].set_title(f"Prediction (t={threshold})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()
    
def plot_training_curves(
    checkpoint_path,
    title="Training Curves",
    device="cpu"
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
