import torch
from torch import nn

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        # 1. Calculate BCE
        bce_loss = self.bce(inputs, targets)
        
        # 2. Calculate Dice
        probs = torch.sigmoid(inputs)
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + 1e-8) / (probs.sum() + targets.sum() + 1e-8)  
        
        # 3. Combine
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 1. Calculate Focal Loss
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        
        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t handles class imbalance (background vs foreground)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # The focal weight scales down the loss for easy, confident predictions
        focal_loss = (alpha_t * (1 - p_t) ** self.gamma * bce_loss).mean()
        
        # 2. Calculate Dice Loss
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + 1e-8) / (probs_flat.sum() + targets_flat.sum() + 1e-8)  
        
        # 3. Combine
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
