"""
Created on Aug 31, 2024.
main_fltl_FL.py

@author: Mahshad Lotfinia <mahshad.lotfinia@rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""

import pdb
import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import timm
import numpy as np
from sklearn import metrics

from config.serde import open_experiment, create_experiment, write_config
from Train_Valid_fltl import Training
from Prediction_fltl import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D, PediCXR_data_loader_2D

import warnings
warnings.filterwarnings('ignore')




def main_train_federated(global_config_path="", valid=False, resume=False, augment=False, experiment_name='name', train_sites=['vindr', 'cxr14', 'chexpert', 'padchest', 'pedi'], pretrained=True, vit=True, dinov2=False, image_size=224, batch_size=30, lr=1e-5):
    """

        Parameters
        ----------
        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    train_loader = []
    valid_loader = []
    weight_loader = []
    loss_function_loader = []
    label_names_loader = []

    for dataset_name in train_sites:

        if dataset_name == 'vindr':
            train_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'padchest':
            train_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
        elif dataset_name == 'pedi':
            train_dataset_model = PediCXR_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
            valid_dataset_model = PediCXR_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

        model_info = params['Network']
        model_info['lr'] = lr
        model_info['batch_size'] = batch_size
        params['Network'] = model_info
        write_config(params, cfg_path, sort_keys=True)

        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels

        loss_function_model = BCEWithLogitsLoss

        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=batch_size,
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)

        train_loader.append(train_loader_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        label_names_loader.append(label_names_model)

        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
        valid_loader.append(valid_loader_model)

    # Changeable network parameters for the global network
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight_model))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight_model), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='resnet50d', pretrained=pretrained)

    trainer = Training(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoints(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model=model, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, vit=vit)




def load_pretrained_timm_model(num_classes=2, model_name='resnet50d', pretrained=False, imgsize=512):

    if model_name == 'resnet50d':
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

    else:
        model = timm.create_model(model_name, num_classes=num_classes, img_size=imgsize, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = True

    return model


def load_pretrained_dinov2(num_classes=2):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.head = torch.nn.Linear(in_features=768, out_features=num_classes)

    for param in model.parameters():
        param.requires_grad = True

    return model






if __name__ == '__main__':
    main_train_federated(global_config_path="/PATH/config/config.yaml", valid=True, resume=True, augment=True,
                         experiment_name='name', train_sites=['vindr', 'cxr14', 'chexpert', 'padchest', 'pedi'])
