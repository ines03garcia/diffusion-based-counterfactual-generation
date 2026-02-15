import os
import json
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve


def evaluate_metrics(labels, predictions, all = False):
    y_true = np.asarray(labels).astype(int)
    y_pred = np.asarray(predictions).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=1)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    if not all:
        return f1, balanced_accuracy

    cm_optimal = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm_optimal.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "specificity": float(spec),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return out    

def print_metrics(metrics, aucroc):
    print(f"\n{'='*20}")
    print(f"Classification Metrics")
    print(f"{'='*20}")
    
    # Performance metrics
    print(f"Accuracy:       {metrics['acc']:.4f}")
    print(f"Balanced Acc:   {metrics['bacc']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"Specificity:    {metrics['specificity']:.4f}")
    print(f"F1-Score:       {metrics['f1']:.4f}")
    print(f"ROC-AUC:        {aucroc:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"Actual  Neg  {int(metrics['tn']):4d}  {int(metrics['fp']):4d}")
    print(f"        Pos  {int(metrics['fn']):4d}  {int(metrics['tp']):4d}")
    print(f"{'='*20}\n")

def save_metrics_json(metrics, args):
    if metrics is not None:
        # Round float metrics to 4 decimal places
        rounded_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                rounded_metrics[key] = round(value, 4)
            else:
                rounded_metrics[key] = value
        
        # Save metrics as JSON
        metrics_path = os.path.join(args.output_dir, f'{args.eval_set}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(rounded_metrics, f, indent=4)
        print(f"\nSaved {args.eval_set} metrics to: {metrics_path}")

def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: true binary labels (shape = [n_samples, n_classes])
        pred: probability estimates of the positive class (shape = [n_samples, n_classes])

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def compute_accuracy(gt, pred):
    return (((pred == gt).sum()) / gt.size(0)).item() * 100


def compute_auprc(gt, pred):
    return average_precision_score(gt, pred)


def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score


# https://www.kaggle.com/code/sohier/probabilistic-f-score
def pfbeta(gt, pred, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def auroc(gt, pred):
    return roc_auc_score(gt, pred)


def calculate_youden_threshold(y_true, y_probs):
    """
    Calculate the optimal threshold using Youden's J statistic.
    
    Youden's J statistic = Sensitivity + Specificity - 1
    The threshold that maximizes this statistic is selected.
    
    Args:
        y_true: Ground truth binary labels (numpy array)
        y_probs: Predicted probabilities (numpy array)
    
    Returns:
        optimal_threshold: Threshold value that maximizes Youden's J statistic
        youden_j: The maximum Youden's J statistic value
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Calculate Youden's J statistic for each threshold
    # J = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    youden_j = tpr - fpr
    
    # Find the optimal threshold
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    max_youden_j = youden_j[optimal_idx]
    
    return optimal_threshold, max_youden_j


def pfbeta_binarized(gt, pred):
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)
