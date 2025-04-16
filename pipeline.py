import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm

from custom_dataset import CustomDataset
from mil_classifier import PatchFeatureExtractor, FeatureExtractor, AttentionClassifier, MILClassifier

from models import dsmil
from models import snuffy

import copy


def prepare_dataset(pos_data_dir, neg_data_dir, batch_size=64):
    """
    Prepares the dataset and dataloaders.

    Args:
        pos_data_dir (str): Directory containing positive samples.
        neg_data_dir (str): Directory containing negative samples.
        batch_size (int): Batch size for dataloaders.

    Returns:
        DataLoader: Dataloader for training data.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CustomDataset(pos_data_dir, neg_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(model : MILClassifier, train_loader, val_loader, epochs=100, device='cuda', log_dir='./logs_mil', best_model_path='./best_model.pth'):
    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    # initialize best losses
    best_losses = dict()
    for epoch in (range(epochs)):
        # Training phase
        model.train()
        # initialize train losses
        train_losses = dict()
        print(f"Epoch {epoch+1}/{epochs}. Training step.")
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

        # Validation phase
        model.eval()
        # initialize validation losses
        val_losses = dict()
        print(f"Validation step.")
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

        # # log validation images to tensorboard
        # val_images = model.val_images(X, y)
        # for name, images in val_images.items():
        #     if len(images) == 0:
        #         continue
        #     writer.add_image(name, make_grid(images.clamp(0, 1)), global_step=epoch)

        # Save best model
        if model.is_better(val_losses, best_losses):
            best_losses = val_losses
            torch.save(model.state_dict(), best_model_path)


def train_dsmil(device="cuda", epochs=360):
    train_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/train/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/train/")
    val_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/val/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/val/")
    # test_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/test/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/test/")

    # Define model, loss, optimizer, and number of epochs
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load("/home/mcloud-ai/Desktop/dc/br/data/mnv4_ssl_224.pth"))
    patch_feature_extractor.requires_grad_(False)

    instnace_classifier = dsmil.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)
    # instnace_classifier.to(device)

    bag_classifier = dsmil.BClassifier(input_size=128, output_class=1, dropout_v=0.0, nonlinear=True, passing_v=False)
    # bag_classifier.to(device)

    mil_net = dsmil.MILNet(instnace_classifier, bag_classifier)
    # mil_net.to(device)

    model = dsmil.BatchMILNet(mil_net)
    model.to(device)

    best_model_path = "best_dsmil_1try.pth"

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_dsmil_1try", 
                best_model_path=best_model_path)


def train_snuffy(device="cuda", epochs=360):
    train_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/train/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/train/")
    val_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/val/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/val/")
    # test_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/test/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/test/")

    # Define model, loss, optimizer, and number of epochs
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load("/home/mcloud-ai/Desktop/dc/br/data/mnv4_ssl_224.pth"))
    patch_feature_extractor.requires_grad_(False)

    instnace_classifier = snuffy.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)
    # instnace_classifier.to(device)
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
    # bag_classifier.to(device)

    mil_net = snuffy.MILNet(instnace_classifier, bag_classifier)
    # mil_net.to(device)

    model = snuffy.BatchMILNet(mil_net)
    model.to(device)

    best_model_path = "best_snuffy_1try.pth"

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_snuffy_1try", 
                best_model_path=best_model_path)


def train_naive(device="cuda", epochs=360):
    # Prepare datasets
    train_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/train/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/train/")
    val_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/val/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/val/")
    test_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/test/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/test/")


    # Define model, loss, optimizer, and number of epochs
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load("/home/mcloud-ai/Desktop/dc/br/data/mnv4_ssl_224.pth"))

    feature_extractor = FeatureExtractor(patch_feature_extractor)
    # feature_extractor.to(device)
    feature_extractor.requires_grad_(False)

    attention_classifier = AttentionClassifier(feature_dim=128, num_heads=8)
    # attention_classifier.to(device)

    model = MILClassifier(feature_extractor, attention_classifier)

    best_model_path = "best_full_model_8heads144patches.pth" 

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_mil_360_8heads144patches", 
                best_model_path=best_model_path)
    # Test the model
    # test_model(model, best_model_path, test_loader)

if __name__ == "__main__":
    train_snuffy()
    # train_dsmil()
    # train_naive()