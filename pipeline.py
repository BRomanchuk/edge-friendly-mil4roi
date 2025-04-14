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


def prepare_dataset(pos_data_dir, neg_data_dir, batch_size=256):
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



def train_autoencoder(device="cuda", epochs=10):
    # Prepare datasets
    train_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/train/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/train/")
    val_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/val/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/val/")
    test_loader = prepare_dataset(pos_data_dir="/home/mcloud-ai/Desktop/dc/br/data/pos_frames_split/test/", neg_data_dir="/home/mcloud-ai/Desktop/dc/br/data/neg_frames_hfps_split/test/")


    # Define model, loss, optimizer, and number of epochs
    patch_feature_extractor = PatchFeatureExtractor()
    patch_feature_extractor.load_state_dict(torch.load("/home/mcloud-ai/Desktop/dc/br/data/mnv4_ssl_224.pth"))

    feature_extractor = FeatureExtractor(patch_feature_extractor)
    feature_extractor.to(device)
    feature_extractor.requires_grad_(False)

    attention_classifier = AttentionClassifier(feature_dim=128, num_heads=2)
    attention_classifier.to(device)

    model = MILClassifier(feature_extractor, attention_classifier)

    best_model_path = "best_full_model.pth" 

    # Train the model
    train_model(model, train_loader, val_loader, epochs, device, log_dir="./logs_mil", 
                best_model_path=best_model_path)
    # Test the model
    # test_model(model, best_model_path, test_loader)

if __name__ == "__main__":
    train_autoencoder()