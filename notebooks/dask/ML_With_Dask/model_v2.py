import torch
import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY

class PrithviSegmentation(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        
        # 1. Use the V2 backbone (300M version)
        # We explicitly set num_frames=1 for standard static segmentation
        self.encoder = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300", 
            pretrained=True,
            num_frames=3  
        )
        
        # 2. Update embed_dim to 1024 (standard for Prithvi-300M / ViT-Large)
        self.embed_dim = 1024
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1), # Increased to 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x shape should be (B, C, H, W)
        # Prithvi V2 expects (B, C, T, H, W). If input is 4D, we add the T dimension.
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Add time dim -> (B, C, 1, H, W)
            
        features = self.encoder(x)
        
        # Handle output list (Terratorch often returns a list of features)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Reshape tokens to spatial grid
        if features.dim() == 3:
            # Remove CLS token if present (check sequence length vs grid size)
            # Prithvi output is usually (B, L, C)
            if features.shape[1] % 2 != 0:
                features = features[:, 1:, :]
                
            B, L, C = features.shape
            H_grid = W_grid = int(L ** 0.5) # Assumes square image
            
            features = features.permute(0, 2, 1).view(B, C, H_grid, W_grid)
        
        return self.decoder(features)
