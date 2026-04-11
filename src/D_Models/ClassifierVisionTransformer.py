import os
import logging
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

from src.config import MODELS_ROOT
from src.E_Aux_Scripts.utils import check_internet_connection

logger = logging.getLogger(__name__)


class VisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, grad_cam=False):
        super(VisionTransformerClassifier, self).__init__()
        
        if pretrained:
            # Check if internet connection is available
            if check_internet_connection():
                logger.info("Internet connection available. Loading pretrained weights online...")
                try:
                    from torchvision.models import ViT_B_16_Weights
                    self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                    logger.info("Successfully loaded pretrained weights from torchvision")
                except Exception as e:
                    logger.error(f"Failed to load weights online: {e}")
                    logger.info("Falling back to local weights...")
                    self._load_local_weights()
            else:
                logger.info("No internet connection. Loading weights from local file...")
                self._load_local_weights()
        else:
            self.vit = vit_b_16(weights=None)
        
        # Replace classifier head
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        self.grad_cam=grad_cam
    
    def _load_local_weights(self):
        """Load pretrained weights from local file"""
        weights_path = os.path.join(MODELS_ROOT, "vit_b_16-c867db91.pth")
        
        if os.path.exists(weights_path):
            self.vit = vit_b_16(weights=None)
            state_dict = torch.load(weights_path, map_location='cpu')
            self.vit.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        else:
            raise ValueError(f"Local weights not found at {weights_path}. "
                           f"Please download them manually or ensure internet connectivity.")
        
    def freeze_layers(self, freeze_layers=0):
        """Freeze the first layers of the model"""
        for param in self.vit.encoder.layers[:freeze_layers].parameters():
            param.requires_grad = False

    def unfreeze_layers(self, current_epoch, total_epochs):
        """Gradually unfreeze layers based on current epoch"""
        if current_epoch == total_epochs // 4:  # Unfreeze after 25% of training
            logger.info("Unfreezing feature layers...")
            for param in self.vit.encoder.layers.parameters():
                param.requires_grad = True
        elif current_epoch == total_epochs // 2:  # Unfreeze after 50% of training
            logger.info("Unfreezing all layers...")
            for param in self.vit.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        if self.grad_cam:
            return self.vit(x)
        return self.vit(x).squeeze(-1)