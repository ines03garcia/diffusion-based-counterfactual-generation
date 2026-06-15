import os
import logging
import torch
import torch.nn as nn
import timm

from src.config import MODELS_ROOT
from src.E_Aux_Scripts.utils import check_internet_connection

logger = logging.getLogger(__name__)


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        grad_cam=False,
        img_size=(1520, 912),
        model_name="vit_base_patch32_224",
        use_checkpointing=True,
    ):
        super(VisionTransformerClassifier, self).__init__()

        self.grad_cam = grad_cam
        self.model_name = model_name
        self.img_size = img_size

        if pretrained:
            if check_internet_connection():
                logger.info(
                    f"Internet connection available. Loading pretrained timm model "
                    f"{model_name} with img_size={img_size}..."
                )

                self.vit = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=img_size,
                    num_classes=num_classes,
                )

                logger.info("Successfully loaded pretrained weights from timm")

            else:
                logger.info("No internet connection. Loading timm model with local weights...")
                self._load_local_weights(num_classes=num_classes)
        else:
            self.vit = timm.create_model(
                model_name,
                pretrained=False,
                img_size=img_size,
                num_classes=num_classes,
            )

        if use_checkpointing and hasattr(self.vit, "set_grad_checkpointing"):
            logger.info("Enabling gradient checkpointing for ViT")
            self.vit.set_grad_checkpointing(True)

    def _load_local_weights(self, num_classes):
        """
        Load local timm-compatible weights.
        """
        weights_path = os.path.join(MODELS_ROOT, f"{self.model_name}.pth")

        self.vit = timm.create_model(
            self.model_name,
            pretrained=False,
            img_size=self.img_size,
            num_classes=num_classes,
        )

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")

            # Handles checkpoints saved as {"state_dict": ...}
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)

            logger.info(f"Loaded local timm weights from {weights_path}")
            logger.info(f"Missing keys: {missing}")
            logger.info(f"Unexpected keys: {unexpected}")
        else:
            raise ValueError(
                f"Local weights not found at {weights_path}. "
                f"For timm, use a timm-compatible checkpoint, not the torchvision ViT checkpoint."
            )

    def freeze_layers(self, freeze_layers=0):
        """
        Freeze the first transformer blocks.

        timm ViT stores transformer blocks in self.vit.blocks.
        """
        if freeze_layers <= 0:
            return

        logger.info(f"Freezing first {freeze_layers} transformer blocks...")

        for block in self.vit.blocks[:freeze_layers]:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, current_epoch, total_epochs):
        """
        Gradually unfreeze layers based on current epoch.
        """
        if current_epoch == total_epochs // 4:
            logger.info("Unfreezing transformer blocks...")
            for param in self.vit.blocks.parameters():
                param.requires_grad = True

        elif current_epoch == total_epochs // 2:
            logger.info("Unfreezing all ViT parameters...")
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.grad_cam:
            return self.vit(x)

        return self.vit(x).squeeze(-1)