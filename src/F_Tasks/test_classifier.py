import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
METADATA_ROOT = os.path.join(PROJECT_ROOT, "data/metadata")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data/images")

from src.E_Aux_Scripts import logger
from src.E_Aux_Scripts.utils import set_seed, seed_worker
from src.E_Aux_Scripts.classifier_helpers import (
	build_model,
	create_transforms,
    get_test_dataset,
    load_checkpoint_weights,
    run_inference,
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve
)


def main():
	parser = create_argparser()
	args = parser.parse_args()

	if args.data_dir is None:
		if args.dataset == "vindr":
			args.data_dir = os.path.join(IMAGES_ROOT, "VinDr-Mammo-Clip-CLAHE-512")
		else:
			args.data_dir = os.path.join(IMAGES_ROOT, "Inbreast_png")

	if args.metadata_path is None:
		if args.dataset == "vindr":
			args.metadata_path = os.path.join(METADATA_ROOT, "resized_df_512.json")
		else:
			args.metadata_path = os.path.join(METADATA_ROOT, "inbreast_test_metadata.csv")

	set_seed(args.seed)

	setup = f"{args.dataset}_{'with_cf' if args.use_counterfactuals else 'no_cf'}"
	log = logger.Logger(experiment_type="Classifiers", sub_experiment_type="test", model_type=args.model_type, setup=setup)
	log.configure_root_logger()
	log.info(f"Logs will be saved to: {log.output_dir}")

	# Save testing args.
	args_output_path = os.path.join(log.output_dir, "test_args.json")
	with open(args_output_path, "w") as f:
		json.dump(vars(args), f, indent=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log.info(f"Using device [{device}] for testing model [{args.model_type}] on dataset [{args.dataset}]")

	_, test_transform = create_transforms(
		augmentation_type="none",
		model_type=args.model_type,
	)

	test_dataset = get_test_dataset(args, test_transform)
	class_dist = test_dataset.get_class_distribution()
	log.info(f"Loaded {len(test_dataset)} test samples with class distribution {class_dist}")

	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		worker_init_fn=lambda x: seed_worker(args, x),
		generator=torch.Generator().manual_seed(args.seed),
	)

	model = build_model(args, device, experiment="test")

	checkpoint_path = args.checkpoint_path
	if not os.path.isabs(checkpoint_path):
		checkpoint_path = os.path.join(MODELS_ROOT, checkpoint_path)
	load_checkpoint_weights(model, checkpoint_path, device)

	probs, preds, targets, image_ids = run_inference(model, test_loader, device)
	metrics, cm, fpr, tpr, roc_auc = compute_metrics(targets, preds, probs)

	metrics["dataset"] = args.dataset
	metrics["model_type"] = args.model_type
	metrics["checkpoint_path"] = checkpoint_path

	cm_path = os.path.join(log.output_dir, "confusion_matrix.png")
	roc_path = os.path.join(log.output_dir, "roc_curve.png")
	save_confusion_matrix(cm, cm_path)
	save_roc_curve(fpr, tpr, roc_auc, roc_path)

	metrics_output_path = os.path.join(log.output_dir, "test_metrics.json")
	with open(metrics_output_path, "w") as f:
		json.dump(metrics, f, indent=2)

	preds_output_path = os.path.join(log.output_dir, "predictions.json")
	predictions_payload = [
		{
			"image_id": image_id,
			"target": int(target),
			"pred": int(pred),
			"prob": float(prob),
		}
		for image_id, target, pred, prob in zip(image_ids, targets, preds, probs)
	]
	with open(preds_output_path, "w") as f:
		json.dump(predictions_payload, f, indent=2)

	log.info("Testing completed successfully")
	log.info(f"Metrics: {json.dumps(metrics, indent=2)}")
	log.info(f"Saved args to: {args_output_path}")
	log.info(f"Saved metrics to: {metrics_output_path}")
	log.info(f"Saved confusion matrix to: {cm_path}")
	log.info(f"Saved ROC curve to: {roc_path}")
	log.info(f"Saved predictions to: {preds_output_path}")


def create_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, choices=["vindr", "inbreast"], default="vindr")
	parser.add_argument("--data_dir", type=str, default=None)
	parser.add_argument("--metadata_path", type=str, default=None)
	parser.add_argument(
		"--testing_category",
		type=str,
		choices=["all", "healthy", "anomalous", "anomalous_with_findings"],
		default="all",
		help="Category of images to include in testing (VinDr only)",
	)
	parser.add_argument("--cf_dir", type=str, default=os.path.join(IMAGES_ROOT, "repaint_results"))
	parser.add_argument(
		"--use_counterfactuals",
		action="store_true",
		default=False,
		help="Whether to include counterfactual examples during test loading (VinDr only)",
	)
	parser.add_argument("--checkpoint_path", type=str, required=True)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--pretrained", action="store_true", default=True)
	parser.add_argument("--seed", type=int, default=0)

	subparsers = parser.add_subparsers(dest="model_type", required=True)

	convnext_parser = subparsers.add_parser("convnext")
	convnext_parser.add_argument("--batch_size", type=int, default=16)

	vit_parser = subparsers.add_parser("vit")
	vit_parser.add_argument("--batch_size", type=int, default=16)

	mammo_clip_parser = subparsers.add_parser("mammo-clip")
	mammo_clip_parser.add_argument("--batch_size", type=int, default=8)
	mammo_clip_parser.add_argument(
		"--clip_chk_pt_path",
		default=os.path.join(MODELS_ROOT, "b5-model-best-epoch-7.tar"),
		type=str,
		help="Path to Mammo-CLIP checkpoint used to initialize encoder",
	)
	mammo_clip_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b5_period_n_lp", type=str)

	fpn_mil_parser = subparsers.add_parser("fpn-mil")
	fpn_mil_parser.add_argument("--batch_size", type=int, default=8)
	fpn_mil_parser.add_argument("--n_class", type=int, default=1)
	fpn_mil_parser.add_argument("--clip_chk_pt_path", default=os.path.join(MODELS_ROOT, "b2-model-best-epoch-10.tar"), type=str)
	fpn_mil_parser.add_argument("--arch", default="upmc_vindr_breast_clip_det_b2_period_n_lp", type=str)
	fpn_mil_parser.add_argument("--img-size", default=[512, 512], type=int, nargs="*")
	fpn_mil_parser.add_argument("--feature_extraction", default="online", type=str)
	fpn_mil_parser.add_argument("--feat_dim", default=352, type=int)
	fpn_mil_parser.add_argument("--patching", action="store_true", default=True)
	fpn_mil_parser.add_argument("--source_image", type=str, default="patches", choices=["patches", "full_image"])
	fpn_mil_parser.add_argument("--patch_size", type=int, default=128)
	fpn_mil_parser.add_argument("--overlap", type=float, nargs="*", default=[0.0])
	fpn_mil_parser.add_argument("--mil_type", default="pyramidal_mil", choices=[None, "instance", "embedding", "pyramidal_mil"], type=str)
	fpn_mil_parser.add_argument("--pooling_type", default="gated-attention", choices=["max", "mean", "attention", "gated-attention", "pma"], type=str)
	fpn_mil_parser.add_argument("--type_mil_encoder", default="mlp", choices=["mlp", "sab", "isab"], type=str)
	fpn_mil_parser.add_argument("--fcl_attention_dim", type=int, default=128)
	fpn_mil_parser.add_argument("--map_prob_func", type=str, default="softmax", choices=["softmax", "sparsemax", "entmax", "alpha_entmax"])
	fpn_mil_parser.add_argument("--fcl_encoder_dim", type=int, default=256)
	fpn_mil_parser.add_argument("--sab_num_heads", type=int, default=4)
	fpn_mil_parser.add_argument("--isab_num_heads", type=int, default=4)
	fpn_mil_parser.add_argument("--pma_num_heads", type=int, default=1)
	fpn_mil_parser.add_argument("--num_encoder_blocks", type=int, default=2)
	fpn_mil_parser.add_argument("--trans_num_inds", type=int, default=20)
	fpn_mil_parser.add_argument("--trans_layer_norm", type=bool, default=False)
	fpn_mil_parser.add_argument("--multi_scale_model", type=str, choices=["fpn", "backbone_pyramid", "msp"], default="fpn")
	fpn_mil_parser.add_argument("--scales", type=int, nargs="*", default=(16, 32, 128))
	fpn_mil_parser.add_argument("--fpn_dim", type=int, default=256)
	fpn_mil_parser.add_argument("--upsample_method", type=str, choices=["bilinear", "nearest"], default="nearest")
	fpn_mil_parser.add_argument("--norm_fpn", type=bool, default=False)
	fpn_mil_parser.add_argument("--deep_supervision", action="store_true", default=True)
	fpn_mil_parser.add_argument("--type_scale_aggregator", type=str, choices=["concatenation", "max_p", "mean_p", "attention", "gated-attention"], default="gated-attention")
	fpn_mil_parser.add_argument("--drop_classhead", type=float, default=0.0)
	fpn_mil_parser.add_argument("--drop_attention_pool", type=float, default=0.0)
	fpn_mil_parser.add_argument("--drop_mha", type=float, default=0.0)
	fpn_mil_parser.add_argument("--fcl_dropout", type=float, default=0.0)
	fpn_mil_parser.add_argument("--lamda", type=float, default=0.0)
	fpn_mil_parser.add_argument("--nested_model", action="store_true", default=False)

	return parser


if __name__ == "__main__":
	main()
