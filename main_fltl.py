"""
Created on Aug 31, 2024.
main_fltl.py

@author: Mahshad Lotfinia <mahshad.lotfinia@rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""

import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
import timm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from config.serde import open_experiment, create_experiment, write_config
from Train_Valid_fltl import Training
from Prediction_fltl import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D, PediCXR_data_loader_2D

import warnings
warnings.filterwarnings('ignore')




def main_train_central_2D(global_config_path="/PATH/FLTLCXR/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='pedi', pretrained=True, vit=True, dinov2=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/FLTLCXR/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment,
                                                   image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment,
                                                      image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False,
                                                      image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment,
                                                   image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment,
                                                      image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False,
                                                      image_size=image_size)
    elif dataset_name == 'pedi':
        train_dataset = PediCXR_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment,
                                                     image_size=image_size)
        valid_dataset = PediCXR_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False,
                                                     image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    if vit:
        if dinov2:
            model = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model = load_pretrained_timm_model(num_classes=len(weight), pretrained=pretrained, imgsize=image_size)
    else:
        model = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d', pretrained=pretrained)

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    if vit:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                      weight_decay=float(params['Network']['weight_decay']))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr),
                                     weight_decay=float(params['Network']['weight_decay']),
                                     amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])



def main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/PATH/FLTLCXR/config/config.yaml",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 experiment1_epoch_num=100, experiment2_epoch_num=100, dataset_name='pedi', vit_1=True, vit_2=True, dinov2_1=False, dinov2_2=False, image_size=224):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'pedi':
        test_dataset = PediCXR_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)

    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters for the global network
    if vit_1:
        if dinov2_1:
            model1 = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model1 = load_pretrained_timm_model(num_classes=len(weight), imgsize=image_size)
    else:
        model1 = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model(model=model1, epoch_num=experiment1_epoch_num)
    pred_array1, target_array1 = predictor1.predict_only(test_loader)

    #########################################
    pred_array = pred_array1.cpu().numpy()
    target_array = target_array1.int().cpu().numpy()

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)

    #########################################


    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list, dataset_name)

    # Changeable network parameters
    if vit_2:
        if dinov2_2:
            model2 = load_pretrained_dinov2(num_classes=len(weight))
        else:
            model2 = load_pretrained_timm_model(num_classes=len(weight), imgsize=image_size)
    else:
        model2 = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d')

    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    predictor2.setup_model(model=model2, epoch_num=experiment2_epoch_num)
    pred_array2, target_array2 = predictor2.predict_only(test_loader)
    AUC_list2 = predictor2.bootstrapper(pred_array2.cpu().numpy(), target_array2.int().cpu().numpy(), index_list, dataset_name)

    pred_array = pred_array2.cpu().numpy()
    target_array = target_array2.int().cpu().numpy()
    df2 = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df2.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df2.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df2.to_csv(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)



    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')


    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)


    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)



def load_pretrained_timm_model(num_classes=2, model_name='resnet50d', pretrained=False, imgsize=224):
    # Load a pre-trained model from config file
    if model_name == 'resnet50d':
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'densenet121':
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
    main_train_central_2D(global_config_path="/PATH/config/config.yaml", valid=True, resume=False, augment=True, experiment_name='name')
