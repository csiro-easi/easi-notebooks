import boto3, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from label_mappers import WorldCoverLabelMapper
from accelerated_io import GeoBatchSpec, BatchAdapter, get_num_samples, make_dali_iterator
from training_helpers import train_one_epoch, evaluate, SegmentationTask, wrap_model_with_fsdp, train_one_epoch_log
from torchmetrics.functional import confusion_matrix as tm_confusion_matrix
from torchmetrics.functional.segmentation import mean_iou as tm_mean_iou
import seaborn as sns
from s3torchconnector import S3Checkpoint


def load_latest_training_log(s3_bucket: str, log_dir_prefix: str):
    """
    log_dir_prefix should be the same prefix you pass as log_dir, e.g. "logs/".
    Assumes keys like: f"{log_dir_prefix}training_logs_YYYYmmdd_HHMMSS.jsonl"
    """
    s3 = boto3.client("s3")

    resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=log_dir_prefix)
    contents = resp.get("Contents", [])
    keys = [o["Key"] for o in contents if o["Key"].endswith(".jsonl") and "training_logs_" in o["Key"]]
    if not keys:
        raise FileNotFoundError(f"No training_logs_*.jsonl found under s3://{s3_bucket}/{log_dir_prefix}")

    latest_key = max(keys)  # timestamp is in filename, so lexicographic works
    obj = s3.get_object(Bucket=s3_bucket, Key=latest_key)
    text = obj["Body"].read().decode("utf-8")

    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    return latest_key, df


def inspect_timing(df: pd.DataFrame):
    cols = ["epoch", "worker_rank", "t_fetch", "t_to_device", "t_step", "timing_samples", "num_batches"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values(["epoch", "worker_rank"]).to_string(index=False))

    # Aggregate across ranks per epoch (mean)
    agg = df.groupby("epoch", as_index=False)[[c for c in ["t_fetch","t_to_device","t_step"] if c in df.columns]].mean()
    print("\nPer-epoch mean timings (seconds):")
    print(agg.to_string(index=False))

    # Simple plot
    plt.figure(figsize=(8,4))
    for c in ["t_fetch","t_to_device","t_step"]:
        if c in agg.columns:
            plt.plot(agg["epoch"], agg[c], marker="o", label=c)
    plt.xlabel("Epoch")
    plt.ylabel("Seconds (avg over sampled steps)")
    plt.title("Timing breakdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_training_history(bucket, userid, project_name):
    """Plot training curves from latest JSONL log file"""
    
    log_dir = f"{userid}/{project_name}/logs/"
    s3 = boto3.client("s3")
    
    # List all JSONL log files for this project
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=log_dir)
    jsonl_keys = [
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith(".jsonl")
    ]
    
    if not jsonl_keys:
        print("No JSONL log files found under", log_dir)
        return
    
    jsonl_keys.sort()
    latest_key = jsonl_keys[-1]
    print("Using log file:", latest_key)
    
    body = s3.get_object(Bucket=bucket, Key=latest_key)["Body"].read().decode("utf-8")
    logs = [json.loads(line) for line in body.splitlines() if line.strip()]
    logs = sorted(logs, key=lambda x: x["epoch"])
    
    epochs = [log["epoch"] for log in logs]
    train_losses = [log["train_loss"] for log in logs]
    val_losses = [log["val_loss"] for log in logs]
    train_accuracies = [log["train_accuracy"] for log in logs]
    val_accuracies = [log["val_accuracy"] for log in logs]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker="o", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker="o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_model(
    test_repo_prefix, s3_bucket, best_model_key, model, test_snapshot_id=None,
    num_classes: int = 11, batch_size: int = 8, device_id: int = 0,
    ignore_index: int = 255, userid=None, project_name="training_test_project",
    show_training_history=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if show_training_history and userid:
        plot_training_history(s3_bucket, userid, project_name)
    # --- Data ---
    label_mapper = WorldCoverLabelMapper(num_classes=num_classes, ignore_index=ignore_index)
    spec = GeoBatchSpec(x_key="features", y_key="labels", patch_hw=(224, 224))
    adapter = BatchAdapter(label_mapper=label_mapper)

    # --- Indices (no Dataset needed) ---
    n = get_num_samples(
        bucket=s3_bucket,
        repo_prefix=test_repo_prefix,
        snapshot_id=test_snapshot_id,
        x_key=spec.x_key,
    )
    test_indices = np.arange(n)

    test_dl = make_dali_iterator(
        base_indices=test_indices,
        batch_size=batch_size,
        device_id=device_id,
        bucket=s3_bucket,
        repo_prefix=test_repo_prefix,
        snapshot_id=test_snapshot_id,
        #device=device,
        shuffle=False,
        spec=spec,
        adapter=adapter,
    )

    # --- Model + checkpoint ---
    model = model.to(device)

    checkpoint_connection = S3Checkpoint(region="ap-southeast-2")
    with checkpoint_connection.reader(best_model_key) as reader:
        checkpoint = torch.load(reader, map_location=device)

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    task = SegmentationTask(
        num_classes=num_classes,
        class_weights=None,
        ignore_index=ignore_index,
    )

    all_preds_list = []
    all_targets_list = []

    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    # --- Loop over test set ---
    with torch.no_grad():
        for batch in test_dl:
            batch = batch[0]  # DALI iterator: [ {'inputs':..., 'labels':...} ]

            images = batch["inputs"].to(device)   # [B, C, T, H, W] or [B, C, H, W]
            labels = batch["labels"].to(device)   # [B, H, W]

            batch_for_task = {"labels": labels}

            logits = model(images)                # [B, num_classes, H, W]
            loss = task.compute_loss(logits, batch_for_task)

            preds = logits.argmax(dim=1)          # [B, H, W]
            valid = labels != ignore_index        # bool mask

            # basic stats on valid pixels only
            correct = (preds[valid] == labels[valid]).sum()
            pixels = valid.sum()

            total_loss += loss.item() * float(pixels)
            total_correct += int(correct)
            total_pixels += int(pixels)

            all_preds_list.append(preds.detach().cpu())
            all_targets_list.append(labels.detach().cpu())

    if total_pixels > 0:
        avg_loss = total_loss / total_pixels
        pixel_acc = total_correct / total_pixels
    else:
        avg_loss = 0.0
        pixel_acc = 0.0

    # --- Stack predictions and labels ---
    all_preds = torch.cat(all_preds_list, dim=0)    # [N, H, W]
    all_targets = torch.cat(all_targets_list, dim=0)  # [N, H, W]

    print("Label min/max:", int(all_targets.min()), int(all_targets.max()))
    print("Unique labels (sample):", torch.unique(all_targets)[:20])

    # Mask out ignore for global metrics
    valid_mask = all_targets != ignore_index
    preds_valid = all_preds[valid_mask]             # [M]
    targets_valid = all_targets[valid_mask]         # [M]

    # --- Mean IoU (overall + per class) ---
    # reshape to [1, H, W] form (mean_iou expects batch/spatial dims)
    preds_iou = preds_valid.view(1, -1, 1)          # [1, M, 1]
    targets_iou = targets_valid.view(1, -1, 1)      # [1, M, 1]

    miou_per_class = tm_mean_iou(
        preds=preds_iou,
        target=targets_iou,
        num_classes=num_classes,
        include_background=True,
        per_class=True,
        input_format="index",
    )  # shape [num_classes]
    miou_per_class = miou_per_class.view(-1)
    miou_overall = torch.nanmean(miou_per_class).item()

    # --- Confusion matrix (valid pixels only) ---
    cm = tm_confusion_matrix(
        preds_valid,
        targets_valid,
        num_classes=num_classes,
        task="multiclass",
    ).float()  # [num_classes, num_classes]

    true_positives = torch.diag(cm)
    support = cm.sum(dim=1)
    per_class_acc = true_positives / (support + 1e-8)

    # === VISUALIZATIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. CONFUSION MATRIX (normalized)
    cm_norm = cm / (cm.sum(dim=1, keepdim=True) + 1e-8)
    sns.heatmap(cm_norm.cpu().numpy(), annot=False, cmap='Blues', 
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                ax=axes[0,0], cbar_kws={'label': 'Normalized Frequency'})
    axes[0,0].set_title('Confusion Matrix (Normalized)')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. PER-CLASS IoU BAR CHART
    valid_classes = torch.isfinite(miou_per_class)
    axes[0,1].bar(range(num_classes), miou_per_class.cpu().numpy(), 
                  color='skyblue', alpha=0.7)
    axes[0,1].set_title('Mean IoU per Class')
    axes[0,1].set_xlabel('Class')
    axes[0,1].set_ylabel('mIoU')
    axes[0,1].set_ylim(0, 1)
    
    # 3. PER-CLASS ACCURACY BAR CHART
    axes[1,0].bar(range(num_classes), per_class_acc.cpu().numpy(), 
                  color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Accuracy per Class')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_ylim(0, 1)
    axes[1,1].axis('off')  # ← THIS FIXES THE BLANK GRAPH

    plt.tight_layout()
    plt.show()
    
    # 4. SAMPLE PREDICTION vs GROUND TRUTH (first image)
    if len(all_preds) > 0:
        sample_idx = 0
        fig_sample, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        ax1.imshow(all_targets[sample_idx].numpy(), cmap='tab20', vmin=0, vmax=num_classes-1)
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        
        # Prediction
        ax2.imshow(all_preds[sample_idx].numpy(), cmap='tab20', vmin=0, vmax=num_classes-1)
        ax2.set_title('Prediction')
        ax2.axis('off')
        
        # Overlay (GT + Pred)
        overlay = 0.5 * all_targets[sample_idx].numpy() + 0.5 * all_preds[sample_idx].numpy()
        ax3.imshow(overlay, cmap='tab20', vmin=0, vmax=num_classes-1)
        ax3.set_title('Overlay (GT + Pred)')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    plt.tight_layout()
    plt.show()
    
    # === SUMMARY TABLE ===
    summary_data = {
        'Metric': ['Loss', 'Pixel Acc', 'mIoU'],
        'Value': [f'{avg_loss:.4f}', f'{pixel_acc:.4f}', f'{miou_overall:.4f}']
    }
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    import pandas as pd
    print(pd.DataFrame(summary_data).to_string(index=False))
    
    return {
        "loss": avg_loss, "pixel_accuracy": pixel_acc, "miou": miou_overall,
        "miou_per_class": miou_per_class, "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }