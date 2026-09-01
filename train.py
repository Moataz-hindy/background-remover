import torch
import torch.optim as optim
from model import BackgroundRemoval
from loss import FocalDiceLoss
from utils import train_step, val_step

def run_training(
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    checkpoint_path="models/best_model.pth",
    model_class=BackgroundRemoval,
    loss_class=FocalDiceLoss
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_class().to(device)
    loss_fn = loss_class()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_iou = 0.0
    
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []

    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_iou, val_dice = val_step(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_iou": val_iou,
                "val_dice": val_dice,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[*] Saved better model with IoU: {best_iou:.4f}")
            
    return {
        "best_iou": best_iou,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious,
        "val_dices": val_dices
    }
