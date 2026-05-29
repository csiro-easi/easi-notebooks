# train_prithvi.py
"""
Distributed training function for Prithvi water segmentation.
Runs on Dask GPU workers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional
import io
import time


class DiceLoss(nn.Module):
    """Dice loss for segmentation - works well for imbalanced classes like water."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, H, W) ground truth labels
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Calculate Dice per class
        dims = (0, 2, 3)  # Reduce over batch and spatial dims
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss."""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        # Mask out ignore_index for dice
        valid_mask = targets != 255
        if valid_mask.sum() > 0:
            dice_loss = self.dice(logits, targets.clamp(0, logits.shape[1] - 1))
        else:
            dice_loss = torch.tensor(0.0, device=logits.device)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> Dict:
    """
    Compute segmentation metrics.
    
    Returns:
        Dictionary with accuracy, IoU per class, mean IoU
    """
    preds = logits.argmax(dim=1)  # (B, H, W)
    
    # Ignore padding (255)
    valid = targets != 255
    preds_valid = preds[valid]
    targets_valid = targets[valid]
    
    if len(targets_valid) == 0:
        return {'accuracy': 0.0, 'mean_iou': 0.0, 'water_iou': 0.0}
    
    # Accuracy
    correct = (preds_valid == targets_valid).float()
    accuracy = correct.mean().item()
    
    # IoU per class
    ious = []
    for cls in range(num_classes):
        pred_cls = preds_valid == cls
        target_cls = targets_valid == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = float('nan')
        ious.append(iou)
    
    # Mean IoU (excluding NaN)
    valid_ious = [x for x in ious if x == x]  # Filter NaN
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'not_water_iou': ious[0] if len(ious) > 0 else 0.0,
        'water_iou': ious[1] if len(ious) > 1 else 0.0,
    }


def train_prithvi_on_worker(
    zarr_path: str,
    model_state: Optional[bytes],
    worker_rank: int,
    world_size: int,
    num_classes: int = 2,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    patch_size: int = 224,
    freeze_backbone_epochs: int = 2,  # Freeze backbone for first N epochs
) -> Dict:
    """
    Training function that runs on a Dask GPU worker.
    
    Args:
        zarr_path: S3 path to training Zarr store
        model_state: Serialized model state (or None to start fresh)
        worker_rank: This worker's rank
        world_size: Total number of workers
        num_classes: Number of classes (2 for water/not-water)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patch_size: Input patch size
        freeze_backbone_epochs: Freeze backbone for first N epochs
    
    Returns:
        Dictionary with trained model state and metrics
    """
    # Import here to ensure available on worker
    from distributed_dataset import RemoteZarrDataset
    from prithvi_model import PrithviWaterSegmentation
    
    print(f"\n{'='*50}")
    print(f"[Worker {worker_rank}/{world_size}] Starting Prithvi training")
    print(f"{'='*50}")
    
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(worker_rank % torch.cuda.device_count())
        print(f"[Worker {worker_rank}] GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"[Worker {worker_rank}] Using CPU")
    
    # --- Dataset ---
    print(f"[Worker {worker_rank}] Loading dataset from {zarr_path}")
    dataset = RemoteZarrDataset(
        zarr_path=zarr_path,
        patch_size=patch_size,
        worker_rank=worker_rank,
        world_size=world_size,
    )
    num_samples = len(dataset)
    print(f"[Worker {worker_rank}] Dataset: {num_samples} patches")
    
    # --- Model ---
    print(f"[Worker {worker_rank}] Creating Prithvi model...")
    model = PrithviWaterSegmentation(
        num_classes=num_classes,
        freeze_backbone=(freeze_backbone_epochs > 0),
    )
    
    # Load pretrained state if provided
    if model_state is not None:
        state_dict = torch.load(io.BytesIO(model_state), map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"[Worker {worker_rank}] Loaded pretrained weights")
    
    model = model.to(device)
    
    # --- Loss and Optimizer ---
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # --- Metrics tracking ---
    metrics = {
        'train_loss': [],
        'accuracy': [],
        'water_iou': [],
        'samples_processed': 0,
        'time_per_epoch': [],
    }
    
    # --- Training Loop ---
    print(f"[Worker {worker_rank}] Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Unfreeze backbone after initial epochs
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
            # Update optimizer to include backbone params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate * 0.1,  # Lower LR for backbone
                weight_decay=0.01
            )
            print(f"[Worker {worker_rank}] Backbone unfrozen, LR reduced")
        
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'accuracy': 0.0, 'water_iou': 0.0}
        n_batches = 0
        
        # Iterate using thread-based batch loading
        for features, labels in dataset.iter_batches(batch_size=batch_size, shuffle=True):
            features = features.unsqueeze(2)
            # Move to device
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward with mixed precision
            with autocast():
                logits = model(features)
                loss = criterion(logits, labels)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            epoch_loss += loss.item()
            
            with torch.no_grad():
                batch_metrics = compute_metrics(logits, labels, num_classes)
                epoch_metrics['accuracy'] += batch_metrics['accuracy']
                epoch_metrics['water_iou'] += batch_metrics['water_iou']
            
            n_batches += 1
            metrics['samples_processed'] += features.shape[0]
            
            # Progress logging
            if n_batches % 20 == 0:
                print(f"[Worker {worker_rank}] Epoch {epoch+1} | "
                      f"Batch {n_batches} | Loss: {loss.item():.4f}")
        
        # Epoch stats
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_metrics['accuracy'] / max(n_batches, 1)
        avg_water_iou = epoch_metrics['water_iou'] / max(n_batches, 1)
        
        metrics['train_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc)
        metrics['water_iou'].append(avg_water_iou)
        metrics['time_per_epoch'].append(epoch_time)
        
        print(f"[Worker {worker_rank}] Epoch {epoch+1}/{epochs} complete:")
        print(f"    Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | Water IoU: {avg_water_iou:.3f}")
        print(f"    Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # --- Save model state ---
    print(f"[Worker {worker_rank}] Training complete, serializing model...")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    return {
        'model_state': buffer.getvalue(),
        'metrics': metrics,
        'worker_rank': worker_rank,
        'final_loss': metrics['train_loss'][-1],
        'final_water_iou': metrics['water_iou'][-1],
    }