import os
import logging
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

from src.config import MODELS_ROOT
from src.Classifiers.aux_scripts.utils import check_internet_connection

logger = logging.getLogger(__name__)


class VisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
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
    
    def forward(self, x):
        return self.vit(x).squeeze(-1) # Remove squeeze for grad cam calculation