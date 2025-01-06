"""
Created on Jan 2, 2025.
utils.py
visualizations

@author: Mahshad Lotfinia <mahshad.lotfinia@rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""
import os.path

import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
import pandas as pd
from sklearn.metrics import roc_curve



def compute_mean_and_ci(data, confidence=0.95):
    """
    """
    mean_val = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean_val, (mean_val - ci, mean_val + ci)


def ROC_curves_imagenet(local_path, FL_path, datasetname):
    local_path_final = os.path.join(local_path, 'predictions_teston_' + datasetname + '.csv')
    local_path_final = local_path_final.replace('pedi_vitb_imagenet_2labels_lr1e5', datasetname + '_vitb_imagenet_2labels_lr1e5')
    FL_path_final = os.path.join(FL_path, 'predictions_teston_' + datasetname + '.csv')
    local_predictions = pd.read_csv(local_path_final)
    FL_predictions = pd.read_csv(FL_path_final)

    try:
        local_fpr_pneumonia, local_tpr_pneumonia, _ = roc_curve(local_predictions['gt_pneumonia'], local_predictions['prob_pneumonia'])
        FL_fpr_pneumonia, FL_tpr_pneumonia, _ = roc_curve(FL_predictions['gt_pneumonia'], FL_predictions['prob_pneumonia'])
    except:
        local_fpr_pneumonia, local_tpr_pneumonia, _ = roc_curve(local_predictions['gt_Pneumonia'], local_predictions['prob_Pneumonia'])
        FL_fpr_pneumonia, FL_tpr_pneumonia, _ = roc_curve(FL_predictions['gt_Pneumonia'], FL_predictions['prob_Pneumonia'])

    try:
        local_fpr_nofinding, local_tpr_nofinding, _ = roc_curve(local_predictions['gt_no_finding'], local_predictions['prob_no_finding'])
        FL_fpr_nofinding, FL_tpr_nofinding, _ = roc_curve(FL_predictions['gt_no_finding'], FL_predictions['prob_no_finding'])
    except:
        local_fpr_nofinding, local_tpr_nofinding, _ = roc_curve(local_predictions['gt_No finding'], local_predictions['prob_No finding'])
        FL_fpr_nofinding, FL_tpr_nofinding, _ = roc_curve(FL_predictions['gt_No finding'], FL_predictions['prob_No finding'])

    plt.figure(figsize=(10, 8))

    # FL
    plt.plot(FL_fpr_pneumonia, FL_tpr_pneumonia, color='red', lw=3, label='FL pneumonia')
    plt.plot(FL_fpr_nofinding, FL_tpr_nofinding, color='blue', lw=3, label='FL no finding')

    # Local
    plt.plot(local_fpr_pneumonia, local_tpr_pneumonia, color='red', lw=3, linestyle=':', label='Local pneumonia')
    plt.plot(local_fpr_nofinding, local_tpr_nofinding, color='blue', lw=3, linestyle=':', label='Local no finding')


    # Plotting the diagonal line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive rate', fontsize=24)
    plt.ylabel('True positive rate', fontsize=24)
    plt.title('ROC curves for individual labels', fontsize=24, loc='center', pad=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # Adjusting ticks to display a single "0" at origin
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=22)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=22)

    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(datasetname + '_imagenet.png', dpi=300, bbox_inches='tight')



def ROC_curves_dinov2(local_path, FL_path, datasetname):
    local_path_final = os.path.join(local_path, 'predictions_teston_' + datasetname + '.csv')
    local_path_final = local_path_final.replace('pedi_vitb_imagenet_2labels_lr1e5', datasetname + '_vitb_imagenet_2labels_lr1e5')
    FL_path_final = os.path.join(FL_path, 'predictions_teston_' + datasetname + '.csv')
    local_predictions = pd.read_csv(local_path_final)
    FL_predictions = pd.read_csv(FL_path_final)

    try:
        local_fpr_pneumonia, local_tpr_pneumonia, _ = roc_curve(local_predictions['gt_pneumonia'], local_predictions['prob_pneumonia'])
        FL_fpr_pneumonia, FL_tpr_pneumonia, _ = roc_curve(FL_predictions['gt_pneumonia'], FL_predictions['prob_pneumonia'])
    except:
        local_fpr_pneumonia, local_tpr_pneumonia, _ = roc_curve(local_predictions['gt_Pneumonia'], local_predictions['prob_Pneumonia'])
        FL_fpr_pneumonia, FL_tpr_pneumonia, _ = roc_curve(FL_predictions['gt_Pneumonia'], FL_predictions['prob_Pneumonia'])

    try:
        local_fpr_nofinding, local_tpr_nofinding, _ = roc_curve(local_predictions['gt_no_finding'], local_predictions['prob_no_finding'])
        FL_fpr_nofinding, FL_tpr_nofinding, _ = roc_curve(FL_predictions['gt_no_finding'], FL_predictions['prob_no_finding'])
    except:
        local_fpr_nofinding, local_tpr_nofinding, _ = roc_curve(local_predictions['gt_No finding'], local_predictions['prob_No finding'])
        FL_fpr_nofinding, FL_tpr_nofinding, _ = roc_curve(FL_predictions['gt_No finding'], FL_predictions['prob_No finding'])

    plt.figure(figsize=(10, 8))

    # FL
    plt.plot(FL_fpr_pneumonia, FL_tpr_pneumonia, color='red', lw=3, label='FL pneumonia')
    plt.plot(FL_fpr_nofinding, FL_tpr_nofinding, color='blue', lw=3, label='FL no finding')

    # Local
    plt.plot(local_fpr_pneumonia, local_tpr_pneumonia, color='red', lw=3, linestyle=':', label='Local pneumonia')
    plt.plot(local_fpr_nofinding, local_tpr_nofinding, color='blue', lw=3, linestyle=':', label='Local no finding')


    # Plotting the diagonal line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive rate', fontsize=24)
    plt.ylabel('True positive rate', fontsize=24)
    plt.title('ROC curves for individual labels', fontsize=24, loc='center', pad=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # Adjusting ticks to display a single "0" at origin
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=22)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=22)

    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(datasetname + '_dinov2.png', dpi=300, bbox_inches='tight')



def age_histograms(file_path, traintest='train', datasetname='vindr', color='green'):
    data = pd.read_csv(file_path)

    # Define the age interval
    age_interval = [0.0001, 100]

    if traintest=='train':
        train_data = data[(data['split'] == 'train') & (data['age'] >= age_interval[0]) & (data['age'] <= age_interval[1])]
    elif traintest=='test':
        train_data = data[(data['split'] == 'test') & (data['age'] >= age_interval[0]) & (data['age'] <= age_interval[1])]

    # Plot the histogram for age
    plt.figure(figsize=(10, 6))
    plt.hist(train_data['age'], bins=30, color=color, edgecolor='black')
    plt.xlabel('Age [years]', fontsize=16)
    plt.ylabel('Frequency [n]', fontsize=16)
    # plt.title(datasetname, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, None)
    plt.xlim(0, None)
    plt.grid(False)  # Remove gridlines

    # Remove the box around the top and right sides of the figure
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # plt.show()
    plt.savefig(datasetname + traintest + '.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    ROC_curves_dinov2(local_path='/PATH',
               FL_path='/PATH',
               datasetname='chexpert')

