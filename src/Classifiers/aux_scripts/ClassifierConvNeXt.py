import os
import logging
import torch
import torch.nn as nn
from torchvision.models import convnext_base

from src.config import MODELS_ROOT
from src.Classifiers.aux_scripts.utils import check_internet_connection

logger = logging.getLogger(__name__)


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtClassifier, self).__init__()
        
        if pretrained:
            # Check if internet connection is available
            if check_internet_connection():
                try:
                    from torchvision.models import ConvNeXt_Base_Weights
                    self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                    logger.info("Successfully loaded pretrained weights from internet")
                except Exception as e:
                    logger.error(f"Failed to download weights from internet: {e}")
                    logger.info("Falling back to local weights...")
                    self._load_local_weights()
            else:
                logger.info("No internet connection detected. Loading from local file...")
                self._load_local_weights()
        else:
            self.convnext = convnext_base(weights=None)
        
        # Replace the classifier head
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)
    
    def _load_local_weights(self):
        """Load pretrained weights from local file"""
        weights_path = os.path.join(MODELS_ROOT, "convnext_base-6075fbad.pth")
        
        if os.path.exists(weights_path):
            self.convnext = convnext_base(weights=None)  # Create model without weights
            state_dict = torch.load(weights_path, map_location='cpu')
            self.convnext.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        else:
            raise ValueError(f"Local weights not found at {weights_path}")
    
    def forward(self, x):
        return self.convnext(x).squeeze(-1)


