import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

from metrics import segmentation_metrics

def val_step(model, data_loader, loss_fn, device):
    val_loss = 0
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "boundary_f1": 0, "boundary_iou": 0}
    num_samples = 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)

            val_pred = model(X)
            probs = torch.sigmoid(val_pred)
            pred = probs > 0.4

            val_loss += loss_fn(val_pred, y).item() * batch_size
            
            metrics = segmentation_metrics(pred, y)
            for k in total_metrics.keys():
                total_metrics[k] += metrics[k].sum().item()
            
            num_samples += batch_size

        val_loss /= num_samples
        for k in total_metrics.keys():
            total_metrics[k] /= num_samples

        print(f"Val loss: {val_loss:.5f} | Val iou: {total_metrics['iou']:.5f} | Val dice: {total_metrics['dice']:.5f} | "
              f"Val bf1: {total_metrics['boundary_f1']:.5f} | Val biou: {total_metrics['boundary_iou']:.5f}")
    return val_loss, total_metrics['iou'], total_metrics['dice']

def evaluate_thresholds(model, data_loader, device):
    model.eval()
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = {}

    with torch.inference_mode():
        for threshold in thresholds:
            total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "boundary_f1": 0, "boundary_iou": 0}
            num_samples = 0

            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                batch_size = X.size(0)

                logits = model(X)
                probs = torch.sigmoid(logits)
                pred = (probs > threshold).float()

                metrics = segmentation_metrics(pred, y)
                for k in total_metrics.keys():
                    total_metrics[k] += metrics[k].sum().item()
                
                num_samples += batch_size

            for k in total_metrics.keys():
                total_metrics[k] /= num_samples

            results[threshold] = total_metrics

            print(
                f"Threshold: {threshold:.1f} | "
                f"IoU: {total_metrics['iou']:.4f} | "
                f"Dice: {total_metrics['dice']:.4f} | "
                f"Prec: {total_metrics['precision']:.4f} | "
                f"Rec: {total_metrics['recall']:.4f} | "
                f"BF1: {total_metrics['boundary_f1']:.4f} | "
                f"BIoU: {total_metrics['boundary_iou']:.4f}"
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

def postprocess_mask(
    tensor_mask,
    apply_morphology=True,
    apply_contour_filling=True,
    apply_largest_component=True,
    apply_smoothing=True
):
    """
    Takes a PyTorch tensor binary mask and applies cleanup operations conditionally.
    """
    mask = tensor_mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    if apply_morphology:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if apply_contour_filling:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled_mask
    if apply_largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    if apply_smoothing:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask / 255.0
