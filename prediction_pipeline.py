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

from pipeline import prepare_dataset, get_train_val_loaders
# from metrics import compute_metrics
from utils import patch_probs_to_probs_matrix, visualize_mask
import matplotlib.pyplot as plt

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
        for X, y, imgs in tqdm(test_loader):
            X = X.to(device)
            y = y.to(device)

            pos_mask = (y.detach().cpu().numpy() == 1)
            if not pos_mask.any():
                continue

            predictions = model.test_step(X)
            predictions = F.sigmoid(predictions)
            predictions = predictions.detach().cpu().numpy()
            # predictions = predictions.reshape(-1, 29, 16)
            predictions = predictions.reshape(-1, 16, 9)

            max_probs = np.array([np.max(p) for p in predictions])
            print(max_probs)
            print(max_probs * y.detach().cpu().numpy())
            print(max_probs * (1-y.detach().cpu().numpy()))
            # break
            pred_matrix = patch_probs_to_probs_matrix(predictions, patch_matrix_size=(9, 16))
            # pred_matrix = patch_probs_to_probs_matrix(predictions, patch_matrix_size=(16, 29))
            transp_pred_matrix = patch_probs_to_probs_matrix(predictions, patch_matrix_size=(16, 9))
            # transp_pred_matrix = patch_probs_to_probs_matrix(predictions, patch_matrix_size=(29, 16))
            pos_matrices = pred_matrix[pos_mask]
            transp_pos_matrices = transp_pred_matrix[pos_mask]
            # neg_matrices = pred_matrix[~pos_mask]

            print("\n\nPositive matrices:")
            print(pos_matrices[0])
            # plot the first positive matrix and image
            pos_imgs = imgs.detach().cpu().numpy()[pos_mask]
            for pos_img, pos_matrix, trans_pos_matrix in zip(pos_imgs, pos_matrices, transp_pos_matrices):
                fig, ax = plt.subplots(2, 1, figsize=(9, 12))
                ax[0].imshow(pos_img)
                ax[1].imshow(pos_matrix, interpolation='nearest')
                # ax[2].imshow(trans_pos_matrix.T, interpolation='nearest')
                ax[0].set_title("Image")
                ax[1].set_title("Probability Map")
                # ax[1].set_title("Probability Map")
                plt.show()

            # plt.imshow(imgs.detach().cpu().numpy()[pos_mask][0])
            # plt.imshow(pos_matrices[0].T, cmap='hot', interpolation='nearest')
            # plt.show()

            print("\n\nNegative matrices:")
            # print(neg_matrices[0])



            # break
            # print(predictions.shape)
            # print([np.max(p.detach().cpu().numpy()) for p in predictions]))
            # break



def predict_dsmil(device="cuda", fe_model='mn4'):
    # test_loader = prepare_dataset(
    #     pos_data_dir=os.path.join(config.POS_DATA_PATH, "test"),
    #     neg_data_dir=os.path.join(config.NEG_DATA_PATH, "test")
    # )
    test_loader = get_train_val_loaders('dp_large100_crop', batch_size=1, ret_img=True)[1]

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
    # patch_feature_extractor.requires_grad_(False)

    instnace_classifier = dsmil.IClassifier(patch_feature_extractor, feature_size=128, output_class=1)

    bag_classifier = dsmil.BClassifier(input_size=128, output_class=1, dropout_v=0.0, nonlinear=True, passing_v=False)

    mil_net = dsmil.MILNet(instnace_classifier, bag_classifier)

    model = dsmil.BatchMILNet(mil_net)
    if fe_model == 'effvit':
        model.load_state_dict(torch.load("artifacts/best_dsmil_effvit.pth", map_location='cuda:0'))
    elif fe_model == 'mn4':
        # model.load_state_dict(torch.load("artifacts/best_maxloss_crit_500esingle_video_dsmil_lr1e3_wd5e4_mn4.pth", map_location='cuda:0'))
        # model.load_state_dict(torch.load("artifacts/best_maxloss_ps128_crit_600esingle_video_dsmil_lr1e3_wd5e4_mn4.pth", map_location='cuda:0'))
        # model.load_state_dict(torch.load("artifacts/hub/best_maxloss_cropped_ps224_crit_600esingle_video_dsmil_lr1e3_wd1e3_mn4.pth", map_location='cuda:0'))
        # model.load_state_dict(torch.load("artifacts/best_maxloss_ps224_crit_600esingle_video_big_dsmil_lr1e3_wd5e4_mn4.pth", map_location=device))
        model.load_state_dict(torch.load("artifacts/best_maxloss_ps224_crit_600edp_large100_crop_dsmil_lr1e3_wd5e4_mn4.pth", map_location=device))
        # Desktop/dc/br/data/edge-friendly-mil4roi/artifacts/best_maxloss_ps224_crit_600esingle_video_big_dsmil_lr1e3_wd5e4_mn4.pth
        # Desktop/dc/br/data/edge-friendly-mil4roi/artifacts/best_maxloss_ps224_crit_600edp_large100_crop_dsmil_lr1e3_wd5e4_mn4.pth
        
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
        neg_data_dir=os.path.join(config.NEG_DATA_PATH, "test"),
        ret_img=True
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
    # predict_pooling(device="cuda", fe_model='mn4')
    predict_dsmil(device="cpu", fe_model='mn4')