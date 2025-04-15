import numpy as np
import cv2
import torch
from torch import nn
from torchvision.ops import nms
import torch.nn.functional as F

import timm


class PatchFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, embedding_dim=512):
        """
        Initializes the Model class. Modifies the ResNet-18 architecture to remove the final layers
        and add a custom projection head.

        Args:
            feature_dim (int): The size of the final feature vector output.
            embedding_dim (int): The size of the embedding vector before the projection head.
        """
        super(PatchFeatureExtractor, self).__init__()

        model = timm.create_model('mobilenetv4_conv_small', pretrained=True)
        embedding_dim = model.classifier.in_features
        model.classifier = nn.Identity()

        self.f = model

        # Projection head: reduces embedding to the desired feature dimension
        self.g = nn.Sequential(
            nn.Linear(embedding_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim, bias=True)
        )

    def forward(self, x):
        """
        Forward pass through the model. This computes the feature vector and the final output.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple: A tuple containing:
                - feature (torch.Tensor): The normalized feature vector (before projection head).
                - out (torch.Tensor): The final output after applying the projection head, normalized.
        """
        # Pass through the ResNet backbone
        x = self.f(x)

        # Flatten the feature map (excluding the batch dimension) before passing through the projection head
        feature = torch.flatten(x, start_dim=1)

        # Apply the projection head
        out = self.g(feature)

        # Normalize both the feature vector and the final output
        return F.normalize(out, dim=-1)

class FeatureExtractor(nn.Module):
    def __init__(self, patch_level_model):
        super(FeatureExtractor, self).__init__()
        self.patch_level_model = patch_level_model
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