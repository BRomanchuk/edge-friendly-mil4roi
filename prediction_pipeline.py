import os
import config

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
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

from pipeline import prepare_dataset
# from metrics import compute_metrics
from utils import patch_probs_to_probs_matrix

import numpy as np

def test_model(
    model: TrainableModel,
    test_loader: DataLoader,
    device='cuda',
    log_dir='./logs_mil',
):
    # initialize tensorboard writer
    # writer = SummaryWriter(log_dir=log_dir)

    # Testing phase
    model.eval()

    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X = X.to(device)
            y = y.to(device)
            predictions = model.test_step(X)
            # predictions = F.sigmoid(predictions)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions.reshape(-1, 16, 9)

            max_probs = np.array([np.max(p) for p in predictions])
            print(max_probs)
            print(max_probs * y.detach().cpu().numpy())
            print(max_probs * (1-y.detach().cpu().numpy()))
            # break
            # pred_matrix = patch_probs_to_probs_matrix(predictions, patch_matrix_size=(16, 9))
            # print(pred_matrix)
            break
            # print(predictions.shape)
            # print([np.max(p.detach().cpu().numpy()) for p in predictions]))
            # break



def predict_dsmil(device="cuda", fe_model='effvit'):
    test_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "test"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "test")
    )

    # Define feature extractor
    if fe_model == 'mn4':
        patch_feature_extractor = PatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.MN4_PATH))
    elif fe_model == 'mn3':
        patch_feature_extractor = MNv3PatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.MN3_PATH))
    elif fe_model == 'effvit':
        patch_feature_extractor = EffViTPatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.VIT_PATH))   
    patch_feature_extractor.requires_grad_(False)

    instnace_classifier = dsmil.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)

    bag_classifier = dsmil.BClassifier(input_size=128, output_class=1, dropout_v=0.0, nonlinear=True, passing_v=False)

    mil_net = dsmil.MILNet(instnace_classifier, bag_classifier)

    model = dsmil.BatchMILNet(mil_net)
    if fe_model == 'effvit':
        model.load_state_dict(torch.load("artifacts/best_dsmil_effvit.pth", map_location='cuda:0'))
    elif fe_model == 'mn4':
        model.load_state_dict(torch.load("artifacts/best_dsmil_wo_betas_e3_5e4_weight_decay.pth", map_location='cuda:0'))
    model.to(device)

    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_dir=None
    )

def predict_pooling(device="cuda", fe_model='effvit'):
    test_loader = prepare_dataset(
        pos_data_dir=os.path.join(config.POS_DATA_PATH, "test"),
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "test")
    )

    # Define feature extractor
    if fe_model == 'mn4':
        patch_feature_extractor = PatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.MN4_PATH))
    elif fe_model == 'mn3':
        patch_feature_extractor = MNv3PatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.MN3_PATH))
    elif fe_model == 'effvit':
        patch_feature_extractor = EffViTPatchFeatureExtractor()
        # patch_feature_extractor.load_state_dict(torch.load(config.VIT_PATH))    
    
    patch_feature_extractor.requires_grad_(False)

    feature_extractor = FeatureExtractor(patch_feature_extractor)
    feature_extractor.requires_grad_(False)

    instance_classifier = InstanceClassifier(feature_dim=128)

    pooling_classifier = PoolingClassifier(instance_classifier, pooling_type="mean")

    model = MILClassifier(feature_extractor, pooling_classifier)
    if fe_model == 'effvit':
        model.load_state_dict(torch.load("artifacts/best_effvit_meanpool_model.pth", map_location='cuda:0'))
    elif fe_model == 'mn4':
        model.load_state_dict(torch.load("artifacts/best_maxpool_mn4.pth", map_location='cuda:0'))
    model.to(device)

    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_dir=None
    )

if __name__ == "__main__":
    predict_pooling(device="cuda", fe_model='mn4')
    # predict_dsmil(device="cuda")