import logging

from torch import nn

from src.D_Models.MammoCLIP.breastclip.model.modules import load_image_encoder, LinearClassifier

logger = logging.getLogger(__name__)


class BreastClipClassifier(nn.Module):
    def __init__(self, args, ckpt, n_class):
        super(BreastClipClassifier, self).__init__()

        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = args.arch.lower()
        if self.arch.endswith("_lp"):
            logger.info("Linear probing selected via arch; image encoder remains frozen.")
            self.freeze_image_encoder()
        else:
            logger.info("Finetuning selected via arch; image encoder isn't frozen.")

        self.classifier = LinearClassifier(feature_dim=self.image_encoder.out_dim, num_class=n_class)
        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() == "resnet152" or self.config["name"].lower() == "resnet101":
                image_features = self.image_encoder(image)
                return image_features
            else:
                input_dict = {"image": image, "breast_clip_train_mode": True}
                image_features, raw_features = self.image_encoder(input_dict)
                self.raw_features = raw_features
                self.pool_features = image_features
                return image_features
        else:
            image_features = self.image_encoder(image)
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features
    
    def freeze_image_encoder(self):
        """Freeze the full Mammo-CLIP image encoder for linear probing."""
        if not hasattr(self, "image_encoder") or self.image_encoder is None:
            logger.info("Mammo-CLIP: no image encoder found to freeze.")
            return 0

        frozen_tensors = 0
        frozen_scalars = 0

        for param in self.image_encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_tensors += 1
                frozen_scalars += param.numel()

        logger.info(
            f"Mammo-CLIP: froze full image encoder "
            f"({frozen_tensors} tensors, {frozen_scalars:,} scalar parameters)."
        )

        return frozen_tensors

    def forward(self, images):
        if self.image_encoder_type.lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)
        # get image features and predict
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits


class MammoClipInputAdapter(nn.Module):
    """Adapt generic dataloader image layout to the Mammo-CLIP classifier input."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def _prepare_inputs(self, images):
        # CNN path expects NCHW.
        if images.dim() == 5 and images.shape[1] == 1 and images.shape[-1] == 3:
            images = images.squeeze(1).permute(0, 3, 1, 2).contiguous()
        elif images.dim() == 4 and images.shape[-1] == 3 and images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        elif images.dim() != 4:
            raise ValueError(f"Unsupported Mammo-CLIP input shape: {tuple(images.shape)}")

        return images

    def forward(self, images):
        return self.base_model(self._prepare_inputs(images))
