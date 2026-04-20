"""PyTorch models for facial emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineCNN(nn.Module):
    """
    Baseline CNN for emotion recognition.
    
    Architecture:
    - 3 Convolutional blocks with batch normalization
    - Max pooling layers
    - Dropout for regularization
    - 2 Fully connected layers
    
    Input: (batch, 1, 48, 48) - grayscale 48x48 images
    Output: (batch, 7) - logits for 7 emotion classes
    """
    
    def __init__(self, num_classes=7):
        super(BaselineCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected
        # After 3 max pooling: 48 -> 24 -> 12 -> 6, so 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AdvancedCNN(nn.Module):
    """
    Advanced CNN for emotion recognition.
    
    Architecture:
    - 4 Convolutional blocks with double convolutions
    - Batch normalization and dropout for regularization
    - Global average pooling
    - 2 Dense layers with batch norm
    
    Input: (batch, 1, 48, 48) - grayscale 48x48 images
    Output: (batch, 7) - logits for 7 emotion classes
    
    Parameters: ~1.5M (more complex than baseline)
    """
    
    def __init__(self, num_classes=7):
        super(AdvancedCNN, self).__init__()
        
        # Conv Block 1 (double conv)
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Conv Block 2 (double conv)
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Conv Block 3 (double conv)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Conv Block 4 (single conv + global avg pool)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout4 = nn.Dropout(0.3)
        
        # Fully Connected
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.global_avgpool(x)
        x = x.reshape(x.size(0), -1)
        
        # Fully Connected
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEmotion(nn.Module):
    """
    ResNet-style architecture for emotion recognition.
    
    Features:
    - Residual connections for training stability
    - Skip connections
    - 4 residual blocks
    - Dropout regularization
    
    Input: (batch, 1, 48, 48)
    Output: (batch, 7)
    """
    
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        # Squeeze: global average pooling
        squeeze = x.mean((2, 3))
        # Excitation: FC layers
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        # Scale
        excitation = excitation.view(batch, channels, 1, 1)
        return x * excitation


class SEResidualBlock(nn.Module):
    """Residual block with Squeeze-Excitation attention."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExciteBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEResNet(nn.Module):
    """SE-ResNet with Squeeze-Excitation attention blocks."""
    
    def __init__(self, num_classes=7):
        super(SEResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(SEResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(SEResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class DenseBlock(nn.Module):
    """Dense block with multiple convolutions."""
    
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
    
    def forward(self, x):
        new_features = F.relu(self.bn1(x))
        new_features = F.relu(self.bn2(self.conv1(new_features)))
        new_features = self.conv2(new_features)
        return torch.cat([x, new_features], 1)


class DenseNet(nn.Module):
    """DenseNet-style architecture for emotion recognition."""
    
    def __init__(self, num_classes=7):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Dense block 1
        self.dense1 = nn.Sequential(
            DenseBlock(32, 12),
            DenseBlock(44, 12),
            DenseBlock(56, 12),
        )
        # Transition 1
        self.trans1_conv = nn.Conv2d(68, 68, kernel_size=1)
        self.trans1_pool = nn.AvgPool2d(2, 2)
        
        # Dense block 2
        self.dense2 = nn.Sequential(
            DenseBlock(68, 12),
            DenseBlock(80, 12),
            DenseBlock(92, 12),
        )
        # Transition 2
        self.trans2_conv = nn.Conv2d(104, 104, kernel_size=1)
        self.trans2_pool = nn.AvgPool2d(2, 2)
        
        # Dense block 3
        self.dense3 = nn.Sequential(
            DenseBlock(104, 12),
            DenseBlock(116, 12),
            DenseBlock(128, 12),
        )
        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(140, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.dense1(x)
        x = self.trans1_conv(x)
        x = self.trans1_pool(x)
        
        x = self.dense2(x)
        x = self.trans2_conv(x)
        x = self.trans2_pool(x)
        
        x = self.dense3(x)
        x = self.global_avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class InceptionModule(nn.Module):
    """Inception module for multi-scale feature extraction."""
    
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 path
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 path
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 path
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # Pool path
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionNet(nn.Module):
    """Inception-style network for emotion recognition."""
    
    def __init__(self, num_classes=7):
        super(InceptionNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Inception modules (channel values tuned for 48x48 input)
        # Input: 24x24x32
        self.inception1 = InceptionModule(32, 16, 8, 16, 4, 8, 8)  # Output: 24x24x48
        self.pool2 = nn.MaxPool2d(2, 2)  # 12x12x48
        
        # Input: 12x12x48
        self.inception2 = InceptionModule(48, 24, 12, 24, 6, 12, 12)  # Output: 12x12x72
        self.pool3 = nn.MaxPool2d(2, 2)  # 6x6x72
        
        # Input: 6x6x72
        self.inception3 = InceptionModule(72, 32, 16, 32, 8, 16, 16)  # Output: 6x6x96
        
        # Global average pooling and classifier
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.inception1(x)
        x = self.pool2(x)
        
        x = self.inception2(x)
        x = self.pool3(x)
        
        x = self.inception3(x)
        x = self.global_avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class MobileNetV2Block(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""
    
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MobileNetV2Block, self).__init__()
        
        hidden_channels = in_channels * expansion_factor
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        
        # Expansion phase (only if expansion_factor > 1)
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                               stride=stride, padding=1, groups=hidden_channels))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Projection phase
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV2(nn.Module):
    """MobileNetV2-style efficient architecture for emotion recognition."""
    
    def __init__(self, num_classes=7):
        super(MobileNetV2, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MobileNetV2 blocks: (in_channels, out_channels, expansion_factor, stride)
        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, 1, 1),
            MobileNetV2Block(16, 24, 6, 2),  # 48 -> 24
            MobileNetV2Block(24, 24, 6, 1),
            MobileNetV2Block(24, 32, 6, 2),  # 24 -> 12
            MobileNetV2Block(32, 32, 6, 1),
            MobileNetV2Block(32, 32, 6, 1),
            MobileNetV2Block(32, 64, 6, 2),  # 12 -> 6
            MobileNetV2Block(64, 64, 6, 1),
            MobileNetV2Block(64, 64, 6, 1),
            MobileNetV2Block(64, 64, 6, 1),
            MobileNetV2Block(64, 96, 6, 1),
            MobileNetV2Block(96, 96, 6, 1),
            MobileNetV2Block(96, 96, 6, 1),
            MobileNetV2Block(96, 160, 6, 1),
        )
        
        # Final layers
        self.conv_final = nn.Conv2d(160, 128, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(128)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn_final(self.conv_final(x)))
        x = self.global_avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResNet50Transfer(nn.Module):
    """
    ResNet50 with transfer learning for emotion recognition.
    Compatible with saved ResNet50_best.pth
    """
    
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(ResNet50Transfer, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify first layer to accept grayscale input (1 channel)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from original conv1 (average the 3 input channels)
        with torch.no_grad():
            self.backbone.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Replace final FC layer with simple 2-layer head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze all backbone parameters except fc layer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class SmallVGG(nn.Module):
    """Small VGG-style CNN for baseline emotion recognition."""
    
    def __init__(self, num_classes=7, dropout_prob=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, 1024), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


class MediumCNN(nn.Module):
    """Medium-sized custom CNN with better capacity than baseline CNN."""
    
    def __init__(self, num_classes=7, dropout_prob=0.5):
        super().__init__()
        # 5 convolutional blocks with increasing channels
        self.features = nn.Sequential(
            # Block 1: 1 -> 64
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 48 -> 24

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 24 -> 12

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # 12 -> 6

            # Block 4: 256 -> 512
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.MaxPool2d(2),  # 6 -> 3

            # Block 5: 512 feature extraction
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((2, 2)),  # 3 -> 2
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, 1024), nn.ReLU(True), nn.BatchNorm1d(1024), nn.Dropout(dropout_prob),
            nn.Linear(1024, 512), nn.ReLU(True), nn.BatchNorm1d(512), nn.Dropout(dropout_prob),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FER_CNN(nn.Module):
    """FER-CNN model for emotion recognition (0.66 accuracy)."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
        return self.classifier(self.features(x))


def get_model(model_name='baseline', num_classes=7, device='cuda'):
    """
    Factory function to get model.
    
    Args:
        model_name: 'baseline', 'advanced', 'resnet', 'seresnet', 'densenet', 'inception', 'mobilenetv2', or 'resnet50_transfer'
        num_classes: Number of emotion classes (default: 7)
        device: 'cuda' or 'cpu' (auto-falls back to cpu if cuda unavailable)
    
    Returns:
        Model on the specified device
    """
    # Fallback to CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    if model_name == 'baseline':
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == 'advanced':
        model = AdvancedCNN(num_classes=num_classes)
    elif model_name == 'resnet':
        model = ResNetEmotion(num_classes=num_classes)
    elif model_name == 'seresnet':
        model = SEResNet(num_classes=num_classes)
    elif model_name == 'densenet':
        model = DenseNet(num_classes=num_classes)
    elif model_name == 'inception':
        model = InceptionNet(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes)
    elif model_name == 'resnet50_transfer':
        model = ResNet50Transfer(num_classes=num_classes, freeze_backbone=True)
    elif model_name =='medium_cnn':
        model = MediumCNN(num_classes=num_classes)
    elif model_name == 'small_vgg':
        model = SmallVGG(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)
