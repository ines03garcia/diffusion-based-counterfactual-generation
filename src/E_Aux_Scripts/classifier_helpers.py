import os
import numpy as np
import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
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

from src.C_Dataset_Handlers.VinDrMammo_dataset import VinDrMammo_dataset
from src.C_Dataset_Handlers.Inbreast_dataset import Inbreast_dataset
from src.E_Aux_Scripts.LOW import LOWLoss

log = logging.getLogger(__name__)

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


def resume_from_checkpoint(checkpoint_path, model, optimizer, device):
	"""Load model and optimizer state from checkpoint"""

	if not os.path.exists(checkpoint_path):
		log.error(f"Checkpoint not found: {checkpoint_path}")
		return 0, float('inf')

	log.info(f"Resuming from checkpoint: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location=device)

	# Load model state
	model.load_state_dict(checkpoint['model_state_dict'])

	# Load optimizer state
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	# Get checkpoint info
	start_epoch = checkpoint['epoch'] + 1
	best_val_loss = checkpoint.get('val_loss', float('inf'))

	log.info(f"Resumed from epoch {checkpoint['epoch']}, at path {checkpoint_path}, best val loss: {best_val_loss:.4f}")

	return start_epoch, best_val_loss

def create_transforms(
	augmentation_type="standard",
	model_type="convnext",
):
	"""Create classifier transforms for ImageNet-based models and Mammo-CLIP."""
	if model_type == "mammo-clip" or model_type == "fpn-mil": # Mammo-CLIP (UPMC) specific values
		normalize = transforms.Normalize(
			mean=[0.3089279, 0.3089279, 0.3089279], 
			std=[0.25053555408335154, 0.25053555408335154, 0.25053555408335154]
		)
	else: # ImageNet values
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
	elif augmentation_type == "mixup":
		# Add mixup augmentation to the standard set of transformations
		train_transform = transforms.Compose([
			transforms.Lambda(lambda img: img.convert("RGB")),
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=False, mixup_alpha=1.0):
	"""Train for one epoch"""
	model.train()
	running_loss = 0.0
	predictions = []
	targets = []
	uses_low_loss = _uses_low_loss(criterion)

	if use_mixup and uses_low_loss:
		raise ValueError("MixUp is not supported with LOWLoss because it requires hard class targets.")

	for batch in tqdm(dataloader, desc="Training"):
		images, labels, _ = batch
		images = images.to(device)
		labels = labels.to(device)

		if use_mixup:
			images, mixed_labels, _ = mixup_batch(images, labels, alpha=mixup_alpha)

		optimizer.zero_grad()
		outputs = model(images)

		if uses_low_loss:
			labels_for_loss = labels.long()
		else:
			labels_for_loss = labels.float()
			if not labels_for_loss.shape == outputs.shape:
				labels_for_loss = labels_for_loss.view(-1, 1)
		
		if use_mixup:
			loss = criterion(outputs, mixed_labels)
		else:
			loss = criterion(outputs, labels_for_loss)
		
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if uses_low_loss:
			preds = torch.argmax(outputs, dim=1).float() # return class index with the highest score
		else:
			preds = (torch.sigmoid(outputs) > 0.5).float()
		
		predictions.extend(preds.cpu().detach().numpy())
		# Consider the original labels when computing training metrics, even if MixUp is used (soft labels are not suitable for metric calculations)
		targets.extend(labels.long().cpu().numpy() if uses_low_loss else labels.float().cpu().numpy())

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
				labels_for_loss = labels.float()

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

			logits = model(images).view(-1)
			probs = torch.sigmoid(logits)
			preds = (probs > 0.5).float()

			all_probs.extend(probs.cpu().numpy().tolist())
			all_preds.extend(preds.cpu().numpy().tolist())
			all_targets.extend(labels.cpu().numpy().tolist())
			all_image_ids.extend(list(image_ids))

	return (
		np.asarray(all_probs, dtype=np.float32),
		np.asarray(all_preds, dtype=np.int32),
		np.asarray(all_targets, dtype=np.int32),
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


def compute_fixed_recall_metrics(targets, probs, target_recall=0.80):
	def to_pct(value):
		return round(float(value) * 100.0, 1)

	targets = np.asarray(targets, dtype=np.int32)
	probs = np.asarray(probs, dtype=np.float32)

	if len(np.unique(targets)) < 2 or int(np.sum(targets)) == 0:
		raise ValueError("Fixed-recall metrics require at least one positive and one negative sample in the test set.")

	precision, recall, thresholds = precision_recall_curve(targets, probs)
	recall_candidates = np.where(recall[1:] >= target_recall)[0]
	if len(recall_candidates) == 0:
		chosen_index = 0
	else:
		chosen_index = int(recall_candidates[-1])

	decision_threshold = float(thresholds[chosen_index])
	preds = (probs >= decision_threshold).astype(np.int32)

	cm = confusion_matrix(targets, preds, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0.0

	metrics = {
		"fixed_recall_target": round(float(target_recall), 3),
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
		

		