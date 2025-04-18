import os
import config

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm
import copy

from custom_dataset import CustomDataset

from models.feature_extractor import PatchFeatureExtractor, \
    MNv3PatchFeatureExtractor, EffViTPatchFeatureExtractor, FeatureExtractor

from models.mil.base import TrainableModel
from models.mil.naive import AttentionClassifier, MILClassifier
from models.mil.naive import InstanceClassifier, PoolingClassifier
from models.mil import dsmil
from models.mil import snuffy


def prepare_dataset(pos_data_dir, neg_data_dir, batch_size=64):
    """
    Prepares the dataset and dataloader.

    Args:
        pos_data_dir (str): Directory containing positive samples.
        neg_data_dir (str): Directory containing negative samples.
        batch_size (int): Batch size for dataloader.

    Returns:
        DataLoader: Dataloader for training data.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CustomDataset(pos_data_dir, neg_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(
    model : TrainableModel,
    train_loader : DataLoader, 
    val_loader : DataLoader, 
    epochs=100, 
    device='cuda', 
    log_dir='./logs_mil', 
    best_model_path='./best_model.pth'
):
    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    # initialize best losses
    best_losses = dict()
    for epoch in (range(epochs)):
        print(f"Epoch {epoch+1}/{epochs}. Training step.")
        # Training phase
        model.train()
        # initialize train losses
        train_losses = dict()
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            # perform training step
            losses = model.train_step(X, y)
            # accumulate losses
            for loss_name, loss in losses.items():
                train_losses[loss_name] = train_losses.get(loss_name, 0) + loss
        # average losses and log to tensorboard
        for loss_name in train_losses:
            train_losses[loss_name] /= len(train_loader)
            writer.add_scalar(f"Loss/{loss_name}", train_losses[loss_name], epoch)        

        print(f"Validation step.")
        # Validation phase
        model.eval()
        # initialize validation losses
        val_losses = dict()
        for X, y in tqdm(val_loader):
            X = X.to(device)
            y = y.to(device)
            # perform validation step
            losses = model.val_step(X, y)
            for loss_name, loss in losses.items():
                val_losses[loss_name] = val_losses.get(loss_name, 0) + loss
        
        # average losses and log to tensorboard
        for loss_name in val_losses:
            val_losses[loss_name] /= len(val_loader)
            writer.add_scalar(f"ValLoss/{loss_name}", val_losses[loss_name], epoch)

        # Save best model
        if model.is_better(val_losses, best_losses):
            best_losses = val_losses
            torch.save(model.state_dict(), best_model_path)


def train_dsmil(device="cuda", epochs=360):
    train_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "train"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "train")
    )
    val_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "val"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "val")
    )

    # Define feature extractor
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load(config.FEATURE_EXTRACTOR_PATH))
    patch_feature_extractor.requires_grad_(False)

    instnace_classifier = dsmil.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)

    bag_classifier = dsmil.BClassifier(input_size=128, output_class=1, dropout_v=0.0, nonlinear=True, passing_v=False)

    mil_net = dsmil.MILNet(instnace_classifier, bag_classifier)

    model = dsmil.BatchMILNet(mil_net)
    model.to(device)

    best_model_path = "./artifacts/best_dsmil_e3_weight_decay.pth"

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./tb_logs/logs_dsmil_e3_weight_decay", 
                best_model_path=best_model_path)


def train_snuffy(device="cuda", epochs=360):
    train_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "train"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "train")
    )
    val_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "val"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "val")
    )

    # Define feature extractor
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load(config.FEATURE_EXTRACTOR_PATH))
    patch_feature_extractor.requires_grad_(False)

    instnace_classifier = snuffy.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)

    feature_size = 128
    encoder_dropout = 0.0
    big_lambda = 200
    random_patch_share = 0.0
    depth = 1
    mlp_multiplier = 4
    num_classes = 1

    c = copy.deepcopy

    attn = snuffy.MultiHeadedAttention(
        h=2,
        d_model=feature_size,
    ).to(device)

    
    ff = snuffy.PositionwiseFeedForward(
        feature_size,
        feature_size * mlp_multiplier,
        'relu',
        encoder_dropout
    ).to(device)
    bag_classifier = snuffy.BClassifier(
        snuffy.Encoder(
            snuffy.EncoderLayer(
                feature_size,
                c(attn),
                c(ff),
                encoder_dropout,
                big_lambda,
                random_patch_share
            ), depth
        ),
        num_classes,
        feature_size
    )

    mil_net = snuffy.MILNet(instnace_classifier, bag_classifier)

    model = snuffy.BatchMILNet(mil_net)
    model.to(device)

    best_model_path = "best_snuffy_1try.pth"

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_snuffy_1try", 
                best_model_path=best_model_path)


def train_naive(device="cuda", epochs=360):
    train_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "train"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "train")
    )
    val_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "val"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "val")
    )

    # Define feature extractor
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load(config.FEATURE_EXTRACTOR_PATH))
    patch_feature_extractor.requires_grad_(False)

    feature_extractor = FeatureExtractor(patch_feature_extractor)
    feature_extractor.requires_grad_(False)

    attention_classifier = AttentionClassifier(feature_dim=128, num_heads=8)

    model = MILClassifier(feature_extractor, attention_classifier)

    best_model_path = "best_full_model_8heads144patches.pth" 

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_mil_360_8heads144patches", 
                best_model_path=best_model_path)


def train_pooling(device="cuda:0", epochs=100, pooling_type="max"):
    train_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "train"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "train")
    )
    val_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "val"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "val")
    )

    # Define feature extractor
    patch_feature_extractor = PatchFeatureExtractor()
    # patch_feature_extractor = EffViTPatchFeatureExtractor()
    # patch_feature_extractor = MNv3PatchFeatureExtractor()
    # patch_feature_extractor.load_state_dict(torch.load())
    patch_feature_extractor.load_state_dict(torch.load(config.FEATURE_EXTRACTOR_PATH))
    patch_feature_extractor.requires_grad_(False)

    feature_extractor = FeatureExtractor(patch_feature_extractor)
    feature_extractor.requires_grad_(False)

    instance_classifier = InstanceClassifier(feature_dim=128)

    pooling_classifier = PoolingClassifier(instance_classifier, pooling_type=pooling_type)

    model = MILClassifier(feature_extractor, pooling_classifier)
    model.to(device)

    best_model_path = f"./artifacts/best_{pooling_type}pool_model.pth"

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir=f"./tb_logs/logs_mil_{pooling_type}pool_1fc", 
                best_model_path=best_model_path)


if __name__ == "__main__":
    train_snuffy()
    # train_dsmil()
    # train_naive()
    # train_pooling(pooling_type="max")