import torch
import torch.nn.functional as F

def get_boundary(mask, kernel_size=3):
    """
    Extracts the boundary of a binary mask using morphological operations (MaxPool).
    mask: (B, 1, H, W) tensor
    """
    # Max pool acts as morphological dilation
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
    # Min pool acts as morphological erosion
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size//2)
    boundary = dilated - eroded
    return boundary

def boundary_f1(pred, target, tolerance=5):
    """
    Computes Boundary F1 score by comparing boundaries with a tolerance.
    """
    pred_b = get_boundary(pred)
    target_b = get_boundary(target)
    
    # Dilate boundaries to allow for tolerance (e.g. 5x5 kernel -> 2 pixel tolerance)
    target_b_dilated = F.max_pool2d(target_b, tolerance, stride=1, padding=tolerance//2)
    pred_b_dilated = F.max_pool2d(pred_b, tolerance, stride=1, padding=tolerance//2)
    
    # True positives for precision
    tp_p = (pred_b * target_b_dilated).sum(dim=(1, 2, 3))
    # True positives for recall
    tp_r = (target_b * pred_b_dilated).sum(dim=(1, 2, 3))
    
    precision = (tp_p + 1e-8) / (pred_b.sum(dim=(1, 2, 3)) + 1e-8)
    recall = (tp_r + 1e-8) / (target_b.sum(dim=(1, 2, 3)) + 1e-8)
    
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return f1

def boundary_iou(pred, target, tolerance=5):
    """
    Computes Boundary IoU (Intersection over Union of dilated boundaries).
    Matches the formulation of Cheng et al. CVPR 2021.
    """
    pred_b = get_boundary(pred)
    target_b = get_boundary(target)
    
    # Dilate boundaries to capture pixels within tolerance
    pred_b_dilated = F.max_pool2d(pred_b, tolerance, stride=1, padding=tolerance//2)
    target_b_dilated = F.max_pool2d(target_b, tolerance, stride=1, padding=tolerance//2)
    
    # Intersection and Union of the boundary regions
    intersection = (pred_b_dilated * target_b_dilated).sum(dim=(1, 2, 3))
    union = torch.max(pred_b_dilated, target_b_dilated).sum(dim=(1, 2, 3))
    
    return (intersection + 1e-8) / (union + 1e-8)

def segmentation_metrics(pred, target, tolerance=5):
    """
    Computes standard segmentation metrics.
    pred: (B, 1, H, W) binary tensor
    target: (B, 1, H, W) binary tensor
    tolerance: Kernel size for boundary matching (5 = 2px radius)
    Returns per-image metrics to allow proper aggregation.
    """
    pred = pred.float()
    target = target.float()
    
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    
    intersection = tp
    union = tp + fp + fn
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    dice = (2 * intersection + 1e-8) / (2 * intersection + fp + fn + 1e-8)
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    
    bf1 = boundary_f1(pred, target, tolerance=tolerance)
    biou = boundary_iou(pred, target, tolerance=tolerance)
    
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "boundary_f1": bf1,
        "boundary_iou": biou
    }
