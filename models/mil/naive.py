import numpy as np
import cv2
import torch
from torch import nn
from torchvision.ops import nms
import torch.nn.functional as F

import timm


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
        self.fc1 = nn.Linear(feature_dim, 1)
        self.fc2 = nn.Linear(144, 1)
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
        # print('features shape 1', features_batch.shape)
        features_batch = features_batch.permute(1, 0, 2)
        # print('features shape 2', features_batch.shape)
        attention_output, z = self.attention(features_batch, features_batch, features_batch)
        # print('z shape', z.shape)
        # print('attention_output shape', attention_output.shape)
        attention_output = attention_output.permute(1, 0, 2)
        # print('att shape', attention_output.shape)
        # flatten attention_output
        # attention_output = attention_output.reshape((attention_output.shape[0], -1))
        # print('att shape', attention_output.shape)
        logits_batch = self.fc1(attention_output)
        # print('logits shape', logits_batch.shape)
        logits_batch = self.sigmoid(logits_batch)
        logits_batch = logits_batch.squeeze(-1)
        prob = self.fc2(logits_batch).squeeze(-1)
        prob = self.sigmoid(prob)
        # print(prob)
        # print('logits shape', logits_batch.shape)
        return prob
    
class MILClassifier(nn.Module):
    def __init__(self, feature_extractor, classifier_model):
        super(MILClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier_model = classifier_model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

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
        self.optimizer.zero_grad()
        logits_batch = self(patches_batch)
        loss = F.binary_cross_entropy(logits_batch, labels)
        loss.backward()
        self.optimizer.step()
        return {"BCE": loss.item()}
    
    def val_step(self, patches_batch, labels):
        """
        Perform a validation step.

        Args:
            patches_batch (torch.Tensor): Batch of image patches.
            labels (torch.Tensor): Corresponding labels for the patches.

        Returns:
            torch.Tensor: Loss value.
        """
        with torch.no_grad():
            logits_batch = self(patches_batch)
            loss = F.binary_cross_entropy(logits_batch, labels)
        return {"BCE": loss}
    
    def is_better(self, current_losses, best_losses):
        if len(best_losses) == 0:
            return True
        return current_losses["BCE"] < best_losses["BCE"]