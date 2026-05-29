# prithvi_model.py
"""
Prithvi segmentation model for water classification.
Fixed to handle:
- List output from backbone (12 transformer layers)
- CLS token removal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PrithviWaterSegmentation(nn.Module):
    """
    Prithvi-based segmentation model for binary water classification.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        decoder_channels: List[int] = [256, 128, 64, 32],
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embedding_dim = 768
        
        self._load_backbone()
        
        if freeze_backbone:
            self._freeze_backbone()
        
        self.decoder = self._build_decoder(decoder_channels)
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def _load_backbone(self):
        """Load Prithvi backbone from terratorch."""
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            
            self.backbone = BACKBONE_REGISTRY.build(
                'terratorch_prithvi_eo_v1_100',
                pretrained=True
            )
            print("✓ Loaded Prithvi EO v1 100M backbone")
            
        except Exception as e:
            print(f"⚠ Could not load Prithvi: {e}")
            raise
    
    def _freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def _build_decoder(self, channels: List[int]) -> nn.Module:
        """Build upsampling decoder."""
        layers = []
        in_ch = self.embedding_dim
        
        for out_ch in channels:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) where C is number of bands
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. Add temporal dimension for Prithvi: (B, C, H, W) -> (B, C, T=1, H, W)
        x_temporal = x.unsqueeze(2)
        
        # 2. Get backbone features - returns list of 12 layer outputs
        backbone_output = self.backbone(x_temporal)
        
        # 3. Take the last layer output: (B, 1025, 768)
        if isinstance(backbone_output, list):
            tokens = backbone_output[-1]
        else:
            tokens = backbone_output
        
        # 4. Remove CLS token (first token): (B, 1025, 768) -> (B, 1024, 768)
        tokens = tokens[:, 2:, :]
        
        # 5. Reshape to spatial feature map
        h_feat = H // self.patch_size  # 512 / 16 = 32
        w_feat = W // self.patch_size  # 512 / 16 = 32
        
        # (B, 1024, 768) -> (B, 768, 1024) -> (B, 768, 32, 32)
        feat_map = tokens.transpose(1, 2).reshape(B, self.embedding_dim, h_feat, w_feat)
        
        # 6. Decode: (B, 768, 32, 32) -> (B, 64, 256, 256)
        decoded = self.decoder(feat_map)
        
        # 7. Segmentation head: (B, 64, 256, 256) -> (B, num_classes, 256, 256)
        logits = self.seg_head(decoded)
        
        # 8. Upsample to original resolution: -> (B, num_classes, 512, 512)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)


def create_prithvi_model(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    device: torch.device = None
) -> PrithviWaterSegmentation:
    """Factory function to create model."""
    model = PrithviWaterSegmentation(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )
    
    if device is not None:
        model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params, {n_trainable:,} trainable")
    
    return model