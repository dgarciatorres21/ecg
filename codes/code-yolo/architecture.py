# --- File: code_v3.py (The Cleaned-Up Version) ---
import torch
import torch.nn as nn

class YOLO_ECG(nn.Module):
    """
    A robust, fully convolutional YOLO-like model for the ECG task.
    """
    def __init__(self, num_classes=13, num_anchors=3):
        super(YOLO_ECG, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # === Backbone ===
        self.backbone = nn.Sequential(
            self._conv_block(3, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # === Detection Head ===
        output_channels = self.num_anchors * (5 + self.num_classes)
        self.head = nn.Conv2d(128, output_channels, kernel_size=1, stride=1, padding=0)

    def _conv_block(self, in_channels, out_channels, kernel_size):
        """Helper function to create a standard Conv->BatchNorm->LeakyReLU block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        
        # Reshape for easier use: [B, C, H, W] -> [B, H, W, C]
        predictions = predictions.permute(0, 2, 3, 1)
        
        batch_size, grid_h, grid_w, _ = predictions.shape
        
        # Reshape to final intuitive format: [B, Grid_H, Grid_W, Anchors, (5 + Classes)]
        return predictions.view(batch_size, grid_h, grid_w, self.num_anchors, 5 + self.num_classes)