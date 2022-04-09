import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from tools.test import Tester


def Trainer(model, dataloader, optimizer, opt, lr_scheduler=None, skip=False):
    if skip:
        return model
    
    epochs = opt.epochs
    device = opt.device
    train_log = opt.train_log
    model_path = opt.model_path

    # start timing
    t0 = time.time()
    best_acc = 0.0
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        num_batches = 0
        is_save = False
        for seq, labels in dataloader:
            # forward, backward
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seq)           # logits
            prob = torch.softmax(logits)  # prob
            loss = F.nll_loss(prob, labels)
            loss.backward()
            optimizer.step()
            # statistic
            running_loss += loss.detach().item()
            num_batches += 1
        total_loss = running_loss/num_batches
        
        # evaluate
        t1 = time.time()
        duration = round((t1 - t0) / (epoch + 1), 2)
        acc = Tester(model=model, dataloader=dataloader, opt=opt, valid=True)

        # save model
        if acc > best_acc:
            torch.save(model.state_dict(), model_path)
            is_save = True
        
        # save message
        acc_msg = '[Valid][{}] Accuracy: total average on Train dataset: {:.2f}. Whether to save better model: '.format(acc * 100, is_save)
        loss_msg = '[Train][{}] Loss: {:.3f}. Average time in one epoch: {} s'.format(epoch+1, total_loss, duration)
        print(loss_msg)
        print(acc_msg)
        with open(train_log, 'a') as f:
            f.write(loss_msg + "\n" + acc_msg + "\n")
        
        # change learning rate through scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    return model
            