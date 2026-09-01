import torch
import json

from model import BackgroundRemoval
from metrics import segmentation_metrics
from utils import postprocess_mask

def evaluate_model(
    model, 
    data_loader, 
    device, 
    threshold, 
    apply_postprocess, 
    boundary_tolerance=5,
    apply_morphology=True,
    apply_contour_filling=True,
    apply_largest_component=True,
    apply_smoothing=True
):
    model.eval()
    total_metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "boundary_f1": 0, "boundary_iou": 0}
    num_samples = 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)

            logits = model(X)
            probs = torch.sigmoid(logits)
            
            if apply_postprocess:
                pred_masks = []
                for i in range(batch_size):
                    mask = (probs[i] > threshold).float()
                    processed = postprocess_mask(
                        mask,
                        apply_morphology=apply_morphology,
                        apply_contour_filling=apply_contour_filling,
                        apply_largest_component=apply_largest_component,
                        apply_smoothing=apply_smoothing
                    )
                    pred_masks.append(torch.tensor(processed, device=device).unsqueeze(0))
                pred = torch.stack(pred_masks)
                if apply_smoothing:
                    pred = (pred > 0.5).float()
            else:
                pred = (probs > threshold).float()

            metrics = segmentation_metrics(pred, y, tolerance=boundary_tolerance)
            for k in total_metrics.keys():
                total_metrics[k] += metrics[k].sum().item()
            
            num_samples += batch_size

    for k in total_metrics.keys():
        total_metrics[k] /= num_samples

    return total_metrics

def run_evaluation(
    test_loader,
    checkpoint_path="models/best_model.pth",
    threshold=0.4,
    postprocess=True,
    boundary_tolerance=5,
    apply_morphology=True,
    apply_contour_filling=True,
    apply_largest_component=True,
    apply_smoothing=True,
    output_json="evaluation_results.json",
    model_class=BackgroundRemoval
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")

    print(f"Evaluating with threshold {threshold} | Postprocess: {postprocess} | Boundary Tolerance: {boundary_tolerance}...")
    metrics = evaluate_model(
        model, 
        test_loader, 
        device, 
        threshold, 
        postprocess, 
        boundary_tolerance,
        apply_morphology=apply_morphology,
        apply_contour_filling=apply_contour_filling,
        apply_largest_component=apply_largest_component,
        apply_smoothing=apply_smoothing
    )
    
    print("\n--- Final Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {output_json}")
    
    return metrics
