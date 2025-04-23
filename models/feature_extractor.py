
import torch
from torch import nn
import torch.nn.functional as F

import timm


class PatchFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        """
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

class MNv3PatchFeatureExtractor(PatchFeatureExtractor):
    def __init__(self, feature_dim=128):
        """
        Args:
            feature_dim (int): The size of the final feature vector output.
            embedding_dim (int): The size of the embedding vector before the projection head.
        """
        super(PatchFeatureExtractor, self).__init__()

        model = timm.create_model('mobilenetv3_small_100', pretrained=True)
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


class EffViTPatchFeatureExtractor(PatchFeatureExtractor):
    def __init__(self, feature_dim=128):
        """
        Args:
            feature_dim (int): The size of the final feature vector output.
            embedding_dim (int): The size of the embedding vector before the projection head.
        """
        super(EffViTPatchFeatureExtractor, self).__init__()

        model = timm.create_model('efficientvit_b1', pretrained=True)
        embedding_dim = model.head.classifier[0].in_features
        model.head.classifier = nn.Identity()

        self.f = model

        # Projection head: reduces embedding to the desired feature dimension
        self.g = nn.Sequential(
            nn.Linear(embedding_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, patch_level_model):
        super(FeatureExtractor, self).__init__()
        self.patch_level_model = patch_level_model
        # self.patch_level_model.eval()
        self.patch_level_model.requires_grad_(False)
    
    def forward(self, patches_batch):
        with torch.no_grad():
            features_batch = [self.patch_level_model(img_patches) for img_patches in patches_batch]
            features_batch = torch.stack(features_batch)
        return features_batch