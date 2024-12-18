import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt, device):
    data_loader = create_dataloader(opt)

    model.eval()
    
    y_true, y_pred = [], []
    for img, label in tqdm(data_loader):
        in_tens = img.to(device)
        if opt.detection_model in ['SAFE', 'FatFormer']:
            output = model(in_tens).softmax(dim=1)[:, 1]
            y_pred.extend(output.flatten().tolist())
        else:
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())


    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, auc
