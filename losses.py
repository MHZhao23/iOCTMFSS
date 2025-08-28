import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss

class CombinedLoss(nn.Module):
    """
    Combine CrossEntropyLoss and DiceLoss weighted
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(mode='multiclass')
        self.dice_loss = DiceLoss(mode='multiclass')

    def forward(self, pred, target):
        return self.focal_loss(pred, target) + self.dice_loss(pred, target)
