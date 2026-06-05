import os
from PIL.Image import Image
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from albumentations import Compose, HorizontalFlip, VerticalFlip, Affine, ElasticTransform, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	ConfusionMatrixDisplay,
	log_loss,
	precision_recall_curve,
	roc_curve,
	auc,
)

from src.C_Dataset_Handlers.VinDrMammo_dataset import VinDrMammo_dataset, _apply_transform
from src.C_Dataset_Handlers.Inbreast_dataset import Inbreast_dataset
from src.E_Aux_Scripts.LOW import LOWLoss

log = logging.getLogger(__name__)

def add_cf_to_batch(images, labels, image_ids, cf_dir, device, transform):
	"""
	Add counterfactuals to batch while maintaining batch size.
	For each image with a CF, add the CF and remove one image from batch.
	Prioritizes removal of healthy images first, then anomalous without CF.
	Returns pair indices so a paired loss can be computed in the same step.
	"""
	batch_size = images.size(0)
	
	# Identify which images have CFs
	has_cf = []
	cf_images_list = []
	
	for i, image_id in enumerate(image_ids):
		cf_path = os.path.join(cf_dir, f"{image_id}.png")
		if os.path.exists(cf_path):
			has_cf.append(i)
			cf_img = Image.open(cf_path)
			cf_img = _apply_transform(transform, cf_img)
			cf_images_list.append(cf_img)
	
	if not has_cf:
		# No CFs available, return batch as is
		empty_pairs = torch.empty((0, 2), dtype=torch.long, device=device)
		return images, labels, empty_pairs
	
	# Build new batch
	num_cfs = len(has_cf)
	removed = 0
	
	# Remove healthy images
	indices_to_remove = []
	for idx in range(batch_size):
		if removed >= num_cfs:
			break
		if labels[idx].item() == 0 and idx not in has_cf:  # Healthy image without CF
			indices_to_remove.append(idx)
			removed += 1
	
	# Remove anomalous without CF if needed
	if removed < num_cfs:
		for idx in range(batch_size):
			if removed >= num_cfs:
				break
			if labels[idx].item() == 1 and idx not in has_cf:  # Anomalous without CF
				indices_to_remove.append(idx)
				removed += 1
	
	# Keep images that are not being removed
	keep_indices = [i for i in range(batch_size) if i not in indices_to_remove]
	kept_images = images[keep_indices]
	kept_labels = labels[keep_indices]
	
	# Add CF images
	cf_tensor = torch.stack(cf_images_list).to(device)
	new_images = torch.cat([kept_images, cf_tensor], dim=0)
	
	# Add CF labels (all 0 for healthy)
	cf_labels = torch.zeros(num_cfs, dtype=labels.dtype, device=device)
	new_labels = torch.cat([kept_labels, cf_labels], dim=0)

	# Build (original_index_in_new_batch, cf_index_in_new_batch) pairs.
	old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
	cf_start = len(keep_indices)
	pairs = []
	for cf_offset, old_idx in enumerate(has_cf):
		orig_new_idx = old_to_new.get(old_idx)
		if orig_new_idx is not None:
			pairs.append([orig_new_idx, cf_start + cf_offset])

	if pairs:
		pair_indices = torch.tensor(pairs, dtype=torch.long, device=device)
	else:
		pair_indices = torch.empty((0, 2), dtype=torch.long, device=device)
	
	return new_images, new_labels, pair_indices


def compute_pair_consistency_loss(outputs, pair_indices, pair_loss_type="kl"):
	"""Compute consistency between original and CF predictions for paired samples."""
	if pair_indices.numel() == 0:
		return outputs.new_tensor(0.0)

	orig_idx = pair_indices[:, 0]
	cf_idx = pair_indices[:, 1]

	if outputs.dim() > 1 and outputs.shape[1] > 1:
		orig_logits = outputs[orig_idx]
		cf_logits = outputs[cf_idx]
		if pair_loss_type == "mse":
			orig_prob = torch.softmax(orig_logits, dim=1)
			cf_prob = torch.softmax(cf_logits, dim=1)
			return F.mse_loss(orig_prob, cf_prob)

		orig_log_prob = F.log_softmax(orig_logits, dim=1)
		cf_prob = F.softmax(cf_logits, dim=1)
		cf_log_prob = F.log_softmax(cf_logits, dim=1)
		orig_prob = F.softmax(orig_logits, dim=1)
		forward = F.kl_div(orig_log_prob, cf_prob, reduction="batchmean")
		backward = F.kl_div(cf_log_prob, orig_prob, reduction="batchmean")
		return 0.5 * (forward + backward)

	orig_logits = outputs[orig_idx].view(-1)
	cf_logits = outputs[cf_idx].view(-1)
	orig_prob_pos = torch.sigmoid(orig_logits)
	cf_prob_pos = torch.sigmoid(cf_logits)

	if pair_loss_type == "mse":
		return F.mse_loss(orig_prob_pos, cf_prob_pos)

	orig_dist = torch.stack([1.0 - orig_prob_pos, orig_prob_pos], dim=1)
	cf_dist = torch.stack([1.0 - cf_prob_pos, cf_prob_pos], dim=1)
	forward = F.kl_div(torch.log(orig_dist + 1e-8), cf_dist, reduction="batchmean")
	backward = F.kl_div(torch.log(cf_dist + 1e-8), orig_dist, reduction="batchmean")
	return 0.5 * (forward + backward)

def mixup_batch(images, labels, alpha=1.0):
	"""Apply MixUp to a batch of images and labels"""
	batch_size = images.size(0)

	# Generate random mixing coefficient
	mix_coef = np.random.beta(alpha, alpha) # If alpha=1.0, this will give a value sampled from a uniform distribution between 0 and 1

	# Random index permutation for mixing
	index = torch.randperm(batch_size, device=images.device)

	# Mix images and labels
	mixed_images = mix_coef * images + (1 - mix_coef) * images[index, :]
	mixed_labels = mix_coef * labels + (1 - mix_coef) * labels[index]

	return mixed_images, mixed_labels, mix_coef

def build_model(args, device, experiment="train"):
	# Prefer explicit --num-classes for downstream models when provided
	num_classes = getattr(args, 'num_classes', None)
	if num_classes is None or args.loss == "low":
		num_classes = 2 if args.loss == "low" else 1

	if args.model_type == "convnext":
		from src.D_Models.ClassifierConvNeXt import ConvNeXtClassifier
		model = ConvNeXtClassifier(
			num_classes=num_classes,
			pretrained=args.pretrained,
		).to(device)
		log.info("ConvNeXt model created and moved to device")
	elif args.model_type == "vit":
		from src.D_Models.ClassifierVisionTransformer import VisionTransformerClassifier
		model = VisionTransformerClassifier(
			num_classes=num_classes,
			pretrained=args.pretrained,
		).to(device)
		log.info("ViT model created and moved to device")
	elif args.model_type == "mammo-clip":
		from src.D_Models.MammoCLIP.Classifiers.model.breast_clip_classifier import BreastClipClassifier, MammoClipInputAdapter
		clip_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
		base_model = BreastClipClassifier(args, ckpt=clip_ckpt, n_class=num_classes)
		model = MammoClipInputAdapter(base_model).to(device)
		log.info(
			f"Mammo-CLIP model created and moved to device using image encoder type [{base_model.get_image_encoder_type()}]"
		)
	elif args.model_type == "fpn-mil":
		from src.D_Models.FPN_MIL.MIL import build_model as build_mil_model, FpnMilInputAdapter
		args.train = True if experiment == "train" else False
		args.n_class = num_classes
		base_model = build_mil_model(args)
		model = FpnMilInputAdapter(base_model).to(device)
		log.info("FPN-MIL model created and moved to device")
	else:
		raise ValueError(f"Unsupported model type: {args.model_type}")
	return model


def _uses_low_loss(criterion):
	return isinstance(criterion, LOWLoss)


def _prepare_binary_targets(labels, outputs):
	"""Match BCE-style target shape to the model output shape."""
	targets = labels.float()
	if targets.shape != outputs.shape:
		targets = targets.view_as(outputs)
	return targets


def create_optimizer(model, args):
	"""
	Creates optimizer and calculates respective learning rates (regardless of whether layers are currently frozen or not).

	Returns:
		optimizer: Configured AdamW optimizer
	"""
	# Differential learning rates
	if args.use_differential_lr:
		backbone_params = []
		classifier_params = []

		# Include ALL parameters (not just trainable ones)
		for name, param in model.named_parameters():
			if 'classifier' in name or 'heads' in name:
				classifier_params.append(param)
			else:
				backbone_params.append(param)

		optimizer = optim.AdamW([
			{'params': backbone_params, 'lr': args.lr * 0.1},
			{'params': classifier_params, 'lr': args.lr},
		], weight_decay=args.weight_decay)

		log.info(f"Using differential learning rates: backbone LR={args.lr * 0.1:.2e}, classifier LR={args.lr:.2e}")
	else:
		optimizer = optim.AdamW(
			model.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)
		log.info(f"Using uniform learning rate: {args.lr:.2e}")

	return optimizer


class LinearWarmupCosineAnnealingLR(LambdaLR):
	"""
	Linear warmup followed by cosine-annealing scheduler.
	Usage: scheduler = LinearWarmupCosineAnnealingLR(optimizer, total_steps, warmup_steps)
	"""

	def __init__(self, optimizer: optim.Optimizer, total_steps: int, warmup_steps: int, last_epoch: int = -1):
		assert warmup_steps < total_steps, "Warmup steps should be less than total steps."
		self.tsteps = total_steps
		self.wsteps = int(warmup_steps)
		super().__init__(optimizer, self._lr_multiplier, last_epoch)

	def _lr_multiplier(self, step: int) -> float:
		if step < self.wsteps and self.wsteps > 0:
			multiplier = step / float(max(1, self.wsteps))
		else:
			cos_factor = (step - self.wsteps) / max(1, (self.tsteps - self.wsteps))
			multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
		return max(0.0, multiplier)


def resume_from_checkpoint(checkpoint_path, model, optimizer, device):
	"""Load model and optimizer state from checkpoint"""

	if not os.path.exists(checkpoint_path):
		log.error(f"Checkpoint not found: {checkpoint_path}")
		return 0, float('inf')

	log.info(f"Resuming from checkpoint: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location=device)

	# Load model state
	state_dict = checkpoint['model_state_dict']
	if isinstance(model, torch.nn.DataParallel):
		try:
			model.module.load_state_dict(state_dict)
		except RuntimeError:
			model.module.load_state_dict(_strip_module_prefix(state_dict))
	else:
		try:
			model.load_state_dict(state_dict)
		except RuntimeError:
			model.load_state_dict(_strip_module_prefix(state_dict))

	# Load optimizer state
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	# Get checkpoint info
	start_epoch = checkpoint['epoch'] + 1
	best_val_loss = checkpoint.get('val_loss', float('inf'))

	log.info(f"Resumed from epoch {checkpoint['epoch']}, at path {checkpoint_path}, best val loss: {best_val_loss:.4f}")

	return start_epoch, best_val_loss

def create_transforms(
	args=None,
	augmentation_type="standard",
	model_type="convnext",
):
	"""Create classifier transforms for ImageNet-based models and Mammo-CLIP.

	If `args` is provided and `model_type` is `mammo-clip` or `fpn-mil`, use
	the upstream albumentations-style transforms.
	"""
	if model_type in ["mammo-clip", "fpn-mil"]:
		train_transform = Compose([
			HorizontalFlip(),
			VerticalFlip(),
			Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
			ElasticTransform(alpha=getattr(args, 'alpha', 10), sigma=getattr(args, 'sigma', 15)),
			Normalize(
				mean=[getattr(args, 'mean', 0.3089279)] * 3,
				std=[getattr(args, 'std', 0.25053555408335154)] * 3,
				max_pixel_value=255.0,
			),
			ToTensorV2(),
		], p=getattr(args, 'p', 1.0))

		val_transform = Compose([
			Normalize(
				mean=[getattr(args, 'mean', 0.3089279)] * 3,
				std=[getattr(args, 'std', 0.25053555408335154)] * 3,
				max_pixel_value=255.0,
			),
			ToTensorV2(),
		])

		return train_transform, val_transform
	
	# ImageNet values for ViT and ConvNeXt
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], 
		std=[0.229, 0.224, 0.225],
	)

	if augmentation_type == "none":
		train_transform = transforms.Compose([
			transforms.Lambda(lambda img: img.convert("RGB")),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			normalize,
		])
	else:
		train_transform = transforms.Compose([
			transforms.Lambda(lambda img: img.convert("RGB")),
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2),
			transforms.ToTensor(),
			normalize,
		])

	val_transform = transforms.Compose([
		transforms.Lambda(lambda img: img.convert("RGB")),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		normalize,
	])

	return train_transform, val_transform

def train_epoch(
	model,
	dataloader,
	criterion,
	optimizer,
	device,
	add_cf_batch=False,
	cf_dir=None,
	transform=None,
	use_mixup=False,
	mixup_alpha=1.0,
	pair_loss_weight=0.0,
	pair_loss_type="kl",
	scheduler=None,
):
	"""Train for one epoch"""
	model.train()
	running_loss = 0.0
	predictions = []
	targets = []
	uses_low_loss = _uses_low_loss(criterion)

	if use_mixup and uses_low_loss:
		raise ValueError("MixUp is not supported with LOWLoss because it requires hard class targets.")
	if pair_loss_weight > 0.0 and use_mixup:
		raise ValueError("Pair consistency loss is not supported with MixUp because pair labels become ambiguous.")

	for batch in tqdm(dataloader, desc="Training"):
		images, labels, image_ids = batch
		images = images.to(device)
		labels = labels.to(device)
		pair_indices = torch.empty((0, 2), dtype=torch.long, device=device)

		if add_cf_batch:  # Pass this as parameter
			images, labels, pair_indices = add_cf_to_batch(images, labels, image_ids, cf_dir, device, transform)

		if use_mixup:
			images, mixed_labels, _ = mixup_batch(images, labels, alpha=mixup_alpha)

		optimizer.zero_grad()
		# Standard full-precision training path (no AMP)
		outputs = model(images)
		if uses_low_loss:
			labels_for_loss = labels.long()
		else:
			labels_for_loss = _prepare_binary_targets(labels, outputs)

		if use_mixup:
			classification_loss = criterion(outputs, mixed_labels)
		else:
			classification_loss = criterion(outputs, labels_for_loss)

		if add_cf_batch and pair_loss_weight > 0.0:
			pair_loss = compute_pair_consistency_loss(outputs, pair_indices, pair_loss_type=pair_loss_type)
			loss = classification_loss + pair_loss_weight * pair_loss
		else:
			loss = classification_loss

		loss.backward()
		optimizer.step()

		# (Training step already performed above)

		running_loss += loss.item()
		if uses_low_loss:
			preds = torch.argmax(outputs, dim=1).float() # return class index with the highest score
		else:
			preds = (torch.sigmoid(outputs) > 0.5).float()
		
		predictions.extend(preds.cpu().detach().numpy())
		# Consider the original labels when computing training metrics, even if MixUp is used (soft labels are not suitable for metric calculations)
		targets.extend(labels.long().cpu().numpy() if uses_low_loss else labels.float().cpu().numpy())

		# Step scheduler per batch if provided (emulates upstream behavior)
		if scheduler is not None:
			try:
				scheduler.step()
			except Exception:
				pass

	epoch_loss = running_loss / len(dataloader)
	epoch_acc = accuracy_score(targets, predictions)
	epoch_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

	return epoch_loss, epoch_acc, epoch_f1


def validate_epoch(model, dataloader, criterion, device):
	"""Validate for one epoch"""
	model.eval()
	running_loss = 0.0
	predictions = []
	targets = []
	uses_low_loss = _uses_low_loss(criterion)

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Validation"):
			images, labels, _ = batch
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)

			if uses_low_loss:
				labels_for_loss = labels.long()
			else:
				labels_for_loss = _prepare_binary_targets(labels, outputs)

			loss = criterion(outputs, labels_for_loss)

			running_loss += loss.item()
			if uses_low_loss:
				preds = torch.argmax(outputs, dim=1).float()
			else:
				preds = (torch.sigmoid(outputs) > 0.5).float()
			predictions.extend(preds.cpu().numpy())
			targets.extend(labels.long().cpu().numpy() if uses_low_loss else labels.float().cpu().numpy())

	epoch_loss = running_loss / len(dataloader)
	f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

	return epoch_loss, f1, predictions, targets


def _strip_module_prefix(state_dict):
	if not any(key.startswith("module.") for key in state_dict.keys()):
		return state_dict
	return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint_weights(model, checkpoint_path, device):
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
		epoch = checkpoint.get("epoch", "unknown")
		log.info(f"Loading model_state_dict from checkpoint (epoch={epoch})")
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
		log.info("Loading raw state_dict checkpoint")
	else:
		raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

	state_dict = _strip_module_prefix(state_dict)
	if isinstance(model, torch.nn.DataParallel):
		missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
	else:
		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

	if missing_keys:
		log.warning(f"Missing keys while loading checkpoint: {missing_keys}")
	if unexpected_keys:
		log.warning(f"Unexpected keys while loading checkpoint: {unexpected_keys}")

	log.info(f"Checkpoint loaded from: {checkpoint_path}")


def get_test_dataset(args, test_transform):
	if args.dataset == "vindr":
		return VinDrMammo_dataset(
			split="test",
			label=args.testing_category,
			cf_dir=args.cf_dir if args.use_counterfactuals else None,
			cv_fold=0,
			data_dir=args.data_dir,
			metadata_path=args.metadata_path,
			transform=test_transform,
		)

	if args.dataset == "inbreast":
		return Inbreast_dataset(
			data_dir=args.data_dir,
			metadata_path=args.metadata_path,
			transform=test_transform,
			split="test",
		)

	raise ValueError(f"Unsupported dataset: {args.dataset}")


def run_inference(model, dataloader, device):
	model.eval()
	all_probs = []
	all_preds = []
	all_targets = []
	all_image_ids = []

	with torch.no_grad():
		for images, labels, image_ids in dataloader:
			images = images.to(device)
			labels = labels.to(device).float().view(-1)

			logits = model(images)
			
			# Handle multi-class output (when using LOWLoss, model outputs 2 classes)
			if logits.dim() > 1 and logits.shape[1] > 1:
				# Multi-class: take argmax, then convert class 1 to probability
				preds_class = torch.argmax(logits, dim=1).float()
				probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
				preds = preds_class
			else:
				# Binary classification: sigmoid
				logits = logits.view(-1)
				probs = torch.sigmoid(logits)
				preds = (probs > 0.5).float()

			all_probs.extend(probs.cpu().numpy().tolist())
			all_preds.extend(preds.cpu().numpy().tolist())
			all_targets.extend(labels.cpu().numpy().tolist())
			all_image_ids.extend(list(image_ids))

	probs_arr = np.asarray(all_probs, dtype=np.float32)
	preds_arr = np.asarray(all_preds, dtype=np.int32)
	targets_arr = np.asarray(all_targets, dtype=np.int32)
	
	return (
		probs_arr,
		preds_arr,
		targets_arr,
		all_image_ids,
	)


def compute_metrics(targets, preds, probs):
	def to_pct(value):
		return round(float(value) * 100.0, 1)

	cm = confusion_matrix(targets, preds, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

	if len(np.unique(targets)) < 2:
		ll = float("nan")
		fpr = None
		tpr = None
		roc_auc = float("nan")
	else:
		ll = log_loss(targets, probs, labels=[0, 1])
		fpr, tpr, _ = roc_curve(targets, probs)
		roc_auc = auc(fpr, tpr)

	metrics = {
		"accuracy": to_pct(float(accuracy_score(targets, preds))),
		"balanced_accuracy": to_pct(float(balanced_accuracy_score(targets, preds))),
		"precision": to_pct(float(precision_score(targets, preds, zero_division=0))),
		"recall": to_pct(float(recall_score(targets, preds, zero_division=0))),
		"f1_score": to_pct(float(f1_score(targets, preds, zero_division=0))),
		"specificity": to_pct(float(specificity)),
		"roc_auc": None if np.isnan(roc_auc) else to_pct(float(roc_auc)),
		"log_loss": None if np.isnan(ll) else round(float(ll), 3),
		"confusion_matrix": {
			"tn": int(tn),
			"fp": int(fp),
			"fn": int(fn),
			"tp": int(tp),
		},
		"num_samples": int(len(targets)),
	}

	return metrics, cm, fpr, tpr, roc_auc


def compute_fixed_specificity_metrics(targets, probs, target_specificity=0.80):
	def to_pct(value):
		return round(float(value) * 100.0, 1)

	targets = np.asarray(targets, dtype=np.int32)
	probs = np.asarray(probs, dtype=np.float32)

	if len(np.unique(targets)) < 2 or int(np.sum(targets)) == 0:
		raise ValueError("Fixed-specificity metrics require at least one positive and one negative sample in the test set.")

	# Get ROC curve data to find threshold for target specificity
	fpr, tpr, thresholds = roc_curve(targets, probs)
	
	# Specificity = 1 - FPR, so find where FPR <= (1 - target_specificity)
	target_fpr = 1.0 - target_specificity
	fpr_candidates = np.where(fpr <= target_fpr)[0]
	
	if len(fpr_candidates) == 0:
		chosen_index = 0
	else:
		chosen_index = int(fpr_candidates[-1])

	decision_threshold = float(thresholds[chosen_index])
	preds = (probs >= decision_threshold).astype(np.int32)

	cm = confusion_matrix(targets, preds, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0.0

	metrics = {
		"fixed_specificity_target": round(float(target_specificity), 3),
		"decision_threshold": round(float(decision_threshold), 6),
		"accuracy": to_pct(float(accuracy_score(targets, preds))),
		"balanced_accuracy": to_pct(float(balanced_accuracy_score(targets, preds))),
		"precision": to_pct(float(precision_at_threshold)),
		"recall": to_pct(float(recall_at_threshold)),
		"f1_score": to_pct(float(f1_score(targets, preds, zero_division=0))),
		"specificity": to_pct(float(specificity)),
		"confusion_matrix": {
			"tn": int(tn),
			"fp": int(fp),
			"fn": int(fn),
			"tp": int(tp),
		},
		"num_samples": int(len(targets)),
	}

	return metrics, cm, preds


def save_confusion_matrix(cm, output_path):
	fig, ax = plt.subplots(figsize=(6, 6))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (0)", "Anomalous (1)"])
	disp.plot(ax=ax, cmap="Blues", colorbar=False)
	ax.set_title("Confusion Matrix")
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def save_roc_curve(fpr, tpr, roc_auc, output_path):
	fig, ax = plt.subplots(figsize=(6, 6))

	if fpr is None or tpr is None:
		ax.text(0.5, 0.5, "ROC unavailable (single class in y_true)", ha="center", va="center")
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
	else:
		ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
		ax.plot([0, 1], [0, 1], color="navy", linewidth=1, linestyle="--")
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.05)
		ax.legend(loc="lower right")

	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC Curve")
	ax.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

def plot_training_metrics(history, exp_dir, validation=True, use_mixup=False):
	"""Create and save plots showing training metrics evolution"""
	plt.style.use('default')
	epochs = range(1, len(history['train_loss']) + 1)
	
	if not use_mixup:
		# F1 plot
		fig_f1 = plt.figure(figsize=(10, 6))
		if 'train_f1' in history:
			plt.plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2)
		if validation and 'val_f1' in history:
			plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
		
		plt.title('F1 Score Evolution', fontweight='bold', fontsize=14)
		plt.xlabel('Epochs')
		plt.ylabel('F1 Score')
		plt.ylim(0, 1)
		plt.legend()
		plt.grid(True, alpha=0.3)
		
		plt.tight_layout()
		f1_plot_path = os.path.join(exp_dir, 'f1_evolution.png')
		plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
		print(f"F1 plot saved to: {f1_plot_path}")
		plt.close(fig_f1)

	# Loss plot
	fig_loss = plt.figure(figsize=(10, 6))
	if 'train_loss' in history:
		plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
	if validation and 'val_loss' in history:
		plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
	
	plt.title('Loss Evolution', fontweight='bold', fontsize=14)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.ylim(0, 1)
	plt.legend()
	plt.grid(True, alpha=0.3)
	
	plt.tight_layout()
	loss_plot_path = os.path.join(exp_dir, 'loss_evolution.png')
	plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
	print(f"Loss plot saved to: {loss_plot_path}")
	plt.close(fig_loss)
		

		