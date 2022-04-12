from cmath import log
import torch
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def Tester(model, dataloader, opt, valid=False):
    train_log = opt.train_log
    device = opt.device

    model.eval()
    total = 0
    correct = 0
    y_pred = []
    y_true = []
    for batch_dict in dataloader:
        # load data and labels
        data = batch_dict['observed_data'].float().to(device)                        # (B, T, F)
        batch_size, frame_size = data.shape[:2]
        tp = batch_dict['observed_tp'].to(device)                                    # (T,)
        tp = tp[None, :, None].float().expand(batch_size, frame_size, 1)             # (B, T, 1)
        mask = batch_dict['observed_mask'].float().to(device)                        # (B, T, F)
        labels = batch_dict['labels'].squeeze(dim=1).long().to(device)                             # (B,)
        # forward
        logits = model(data, tp, mask)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        correct += (pred == labels).sum()
        total += pred.shape[0]
        # # append
        # y_pred.append(prob[torch.arange(batch_size).to(device), labels].detach().cpu().numpy())
        # y_true.append(labels.unsqueeze(dim=1).detach().cpu().numpy())
    acc = correct / total
    # y_pred = np.concatenate(y_pred, axis=0)
    # y_true = np.concatenate(y_true, axis=0)
    # roc = roc_auc_score(y_true, y_pred)
    if not valid:
        acc_msg = '[Test] Accuracy: total average on Test dataset: {:.2f}.'.format(acc * 100)
        print(acc_msg)
        with open(train_log, 'a') as f:
            f.write(acc_msg + "\n")
    return acc
        