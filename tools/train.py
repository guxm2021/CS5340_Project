import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from tools.test import Tester


def Trainer(model, train_dataloader, valid_dataloader, optimizer, criterion, opt, lr_scheduler=None, skip=False):
    if skip:
        return model
    
    epochs = opt.epochs
    device = opt.device
    train_log = opt.train_log
    model_path = opt.model_path

    # start timing
    t0 = time.time()
    best_acc = 0.0
    best_roc = 0.0 
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        num_batches = 0
        is_save = False
        for batch_dict in train_dataloader:
            # load data and labels
            data = batch_dict['observed_data'].float().to(device)                        # (B, T, F)
            batch_size, frame_size = data.shape[:2]
            tp = batch_dict['observed_tp'].to(device)                                    # (T,)
            tp = tp[None, :, None].float().expand(batch_size, frame_size, 1)             # (B, T, 1)
            mask = batch_dict['observed_mask'].float().to(device)                        # (B, T, F)
            labels = batch_dict['labels'].squeeze(dim=1).long().to(device)                             # (B,)
            # forward, backward
            optimizer.zero_grad()
            logits = model(data, tp, mask)                   # logits
            log_prob = F.log_softmax(logits, dim=-1)         # log-probability
            loss = F.nll_loss(log_prob, labels)
            loss.backward()
            optimizer.step()
            # statistic
            running_loss += loss.detach().item()
            num_batches += 1
            # print(num_batches)
        total_loss = running_loss/num_batches
        
        # evaluate
        t1 = time.time()
        duration = round((t1 - t0) / (epoch + 1), 2)
        acc = Tester(model=model, dataloader=valid_dataloader, opt=opt, valid=True)

        # save model
        if acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = acc
            is_save = True
        # if roc > best_roc:
        #     torch.save(model.state_dict(), model_path)
        #     best_roc = roc
        #     is_save = True
        
        # save message
        acc_msg = '[Valid][{}] Accuracy: total average on Valid dataset: {:.2f}. Whether to save better model: {}'.format(epoch+1, acc * 100, is_save)
        loss_msg = '[Train][{}] Loss: {:.3f}. Average time in one epoch: {} s'.format(epoch+1, total_loss, duration)
        print(loss_msg)
        print(acc_msg)
        with open(train_log, 'a') as f:
            f.write(loss_msg + "\n" + acc_msg + "\n")
        
        # change learning rate through scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    return model
            