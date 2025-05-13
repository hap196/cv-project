import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self, num_classes=150, backbone="resnet50", pretrained=True):
        """
        Fully Convolutional Network for semantic segmentation
        
        Args:
            num_classes (int): Number of segmentation classes
            backbone (str): Backbone network ("resnet18", "resnet34", "resnet50", "vgg16")
            pretrained (bool): Whether to use pretrained weights
        """
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Initialize backbone
        if backbone == "resnet18":
            base_model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        elif backbone == "resnet34": 
            base_model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 2048
        elif backbone == "vgg16":
            base_model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract backbone layers
        if "resnet" in backbone:
            self.stage1 = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            )  # 1/4
            self.stage2 = base_model.layer1  # 1/4
            self.stage3 = base_model.layer2  # 1/8
            self.stage4 = base_model.layer3  # 1/16
            self.stage5 = base_model.layer4  # 1/32
        elif backbone == "vgg16":
            self.stage1 = base_model.features[:5]   # 1/2
            self.stage2 = base_model.features[5:10]  # 1/4
            self.stage3 = base_model.features[10:17]  # 1/8
            self.stage4 = base_model.features[17:24]  # 1/16
            self.stage5 = base_model.features[24:]  # 1/32
        
        # FCN specific layers
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        
        # Skip connections for FCN32, FCN16, FCN8
        if "resnet" in backbone:
            # For FCN16s
            self.score_pool4 = nn.Conv2d(feature_dim // 2, num_classes, kernel_size=1)
            # For FCN8s
            self.score_pool3 = nn.Conv2d(feature_dim // 4, num_classes, kernel_size=1)
        elif backbone == "vgg16":
            # For FCN16s
            self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
            # For FCN8s
            self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()
        
        # Forward through backbone stages
        x = self.stage1(x)
        x = self.stage2(x)
        
        # Save stage3 output for FCN8s
        x3 = self.stage3(x)
        
        # Save stage4 output for FCN16s
        x4 = self.stage4(x3)
        
        # Final stage
        x5 = self.stage5(x4)
        
        # FCN32s: Up-sample the final output directly to input size
        score = self.classifier(x5)
        score_32s = F.interpolate(score, size=input_size[2:], mode='bilinear', align_corners=True)
        
        # FCN16s: Add up-sampled score with pool4 prediction
        score_pool4 = self.score_pool4(x4)
        score_16s = F.interpolate(score, size=score_pool4.size()[2:], mode='bilinear', align_corners=True)
        score_16s = score_16s + score_pool4
        score_16s = F.interpolate(score_16s, size=input_size[2:], mode='bilinear', align_corners=True)
        
        # FCN8s: Add up-sampled score with pool3 prediction
        score_pool3 = self.score_pool3(x3)
        score_8s = F.interpolate(score_16s, size=score_pool3.size()[2:], mode='bilinear', align_corners=True)
        score_8s = score_8s + score_pool3
        score_8s = F.interpolate(score_8s, size=input_size[2:], mode='bilinear', align_corners=True)
        
        # Return FCN8s prediction (the best one)
        return score_8s

    def get_backbone_params(self):
        modules = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    
    def get_classifier_params(self):
        modules = [self.classifier, self.score_pool3, self.score_pool4]
        for i in range(len(modules)):
            for m in modules[i].parameters():
                if m.requires_grad:
                    yield m 