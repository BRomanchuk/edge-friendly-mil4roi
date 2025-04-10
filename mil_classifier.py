import numpy as np
import cv2
import torch
from torch import nn
from torchvision.ops import nms
import torch.nn.functional as F



class FeatureExtractor(nn.Module):
    def __init__(self, patch_level_model, patch_level_model_path):
        super(FeatureExtractor, self).__init__()
        self.patch_level_model = patch_level_model
        self.patch_level_model.load_state_dict(torch.load(patch_level_model_path))
        self.patch_level_model.eval()
        self.patch_level_model.requires_grad_(False)
    
    def forward(self, patches_batch):
        with torch.no_grad():
            features_batch = [self.patch_level_model(img_patches) for img_patches in patches_batch]
            features_batch = torch.stack(features_batch)
        return features_batch
    

class AttentionClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8):
        """
        Initialize the AttentionClassifier.

        Args:
            feature_dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
        """
        super(AttentionClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.fc = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features_batch):
        """
        Forward pass through the attention classifier.

        Args:
            features_batch (torch.Tensor): Batch of features.

        Returns:
            torch.Tensor: Logits for each patch.
        """
        # Reshape features for multi-head attention
        features_batch = features_batch.permute(1, 0, 2)
        attention_output, _ = self.attention(features_batch, features_batch, features_batch)
        attention_output = attention_output.permute(1, 0, 2)
        logits_batch = self.fc(attention_output)
        logits_batch = self.sigmoid(logits_batch)
        logits_batch = logits_batch.squeeze(-1)
        return logits_batch
    
class MILClassifier(nn.Module):
    def __init__(self, feature_extractor, classifier_model, patch_level_model_path):
        super(MILClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier_model = classifier_model
        self.patch_level_model_path = patch_level_model_path

    def forward(self, patches_batch):
        features_batch = self.feature_extractor(patches_batch)
        logits_batch = self.classifier_model(features_batch)
        return logits_batch
    
    def train_step(self, patches_batch, labels):
        """
        Perform a training step.

        Args:
            patches_batch (torch.Tensor): Batch of image patches.
            labels (torch.Tensor): Corresponding labels for the patches.

        Returns:
            torch.Tensor: Loss value.
        """
        logits_batch = self(patches_batch)
        loss = F.binary_cross_entropy(logits_batch, labels)
        return loss